# src/device_manager.py
import torch
import threading
import psutil
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class DeviceInfo:
    """Device information data class"""
    device_type: str  # 'cuda' or 'cpu'
    device_id: int    # GPU ID for CUDA, always 0 for CPU
    total_memory: int # Total memory in bytes
    available_memory: int # Available memory in bytes
    utilization: float # 0.0 to 1.0
    is_available: bool


class DeviceManager:
    """
    Device manager responsible for optimizing GPU and CPU resource allocation
    Supports multi-device concurrent inference and intelligent load balancing
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._devices: Dict[str, DeviceInfo] = {}
            self._device_locks: Dict[str, threading.RLock] = {}
            self._allocation_history: List[Tuple[str, float]] = []
            self._monitoring_active = False
            self._monitor_thread: Optional[threading.Thread] = None
            self._update_interval = 1.0  # seconds
            self._initialized = True
            self._discover_devices()
            self._start_monitoring()
    
    def _discover_devices(self):
        """Discover and initialize available devices"""
        # CPU device
        cpu_memory = psutil.virtual_memory().total
        self._devices['cpu'] = DeviceInfo(
            device_type='cpu',
            device_id=0,
            total_memory=cpu_memory,
            available_memory=cpu_memory,
            utilization=0.0,
            is_available=True
        )
        self._device_locks['cpu'] = threading.RLock()
        
        # CUDA devices
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                device_name = f'cuda:{i}'
                try:
                    # Get GPU memory information
                    torch.cuda.set_device(i)
                    total_memory = torch.cuda.get_device_properties(i).total_memory
                    
                    self._devices[device_name] = DeviceInfo(
                        device_type='cuda',
                        device_id=i,
                        total_memory=total_memory,
                        available_memory=total_memory,
                        utilization=0.0,
                        is_available=True
                    )
                    self._device_locks[device_name] = threading.RLock()
                    print(f"Discovered CUDA device: {device_name} (Memory: {total_memory / 1024**3:.1f}GB)")
                    
                except Exception as e:
                    print(f"Error initializing CUDA device {i}: {e}")
        
        print(f"Device discovery completed: {list(self._devices.keys())}")
    
    def _start_monitoring(self):
        """Start device monitoring thread"""
        if not self._monitoring_active:
            self._monitoring_active = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_devices,
                daemon=True,
                name="DeviceMonitor"
            )
            self._monitor_thread.start()
    
    def _monitor_devices(self):
        """Monitor device usage"""
        while self._monitoring_active:
            try:
                # Update CPU usage
                if 'cpu' in self._devices:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    available_memory = psutil.virtual_memory().available
                    with self._device_locks['cpu']:
                        self._devices['cpu'].utilization = cpu_percent / 100.0
                        self._devices['cpu'].available_memory = available_memory
                
                # Update CUDA device usage
                for device_name in self._devices:
                    if device_name.startswith('cuda:'):
                        device_info = self._devices[device_name]
                        device_id = device_info.device_id
                        
                        try:
                            torch.cuda.set_device(device_id)
                            allocated = torch.cuda.memory_allocated(device_id)
                            cached = torch.cuda.memory_reserved(device_id)
                            total = device_info.total_memory
                            
                            with self._device_locks[device_name]:
                                device_info.available_memory = total - allocated
                                device_info.utilization = allocated / total if total > 0 else 0.0
                                
                        except Exception as e:
                            print(f"Error monitoring device {device_name}: {e}")
                            with self._device_locks[device_name]:
                                device_info.is_available = False
                
                time.sleep(self._update_interval)
                
            except Exception as e:
                print(f"Device monitoring error: {e}")
                time.sleep(self._update_interval)
    
    def get_best_device(self, memory_requirement: int = 0, prefer_cuda: bool = True) -> str:
        """
        Select the best device based on current load and memory requirements
        
        Args:
            memory_requirement: Required memory size (bytes)
            prefer_cuda: Whether to prefer CUDA devices
            
        Returns:
            Best device name (e.g., 'cuda:0' or 'cpu')
        """
        best_device = 'cpu'  # Default fallback to CPU
        best_score = float('-inf')
        
        devices_to_check = list(self._devices.keys())
        if prefer_cuda:
            # Prioritize checking CUDA devices
            cuda_devices = [d for d in devices_to_check if d.startswith('cuda:')]
            cpu_devices = [d for d in devices_to_check if d == 'cpu']
            devices_to_check = cuda_devices + cpu_devices
        
        for device_name in devices_to_check:
            device_info = self._devices[device_name]
            
            if not device_info.is_available:
                continue
            
            # Check if memory is sufficient
            if device_info.available_memory < memory_requirement:
                continue
            
            # Calculate device score (considering utilization and available memory)
            memory_score = device_info.available_memory / device_info.total_memory
            utilization_penalty = device_info.utilization
            
            # CUDA devices get extra score
            device_type_bonus = 2.0 if device_name.startswith('cuda:') and prefer_cuda else 1.0
            
            score = (memory_score - utilization_penalty) * device_type_bonus
            
            if score > best_score:
                best_score = score
                best_device = device_name
        
        print(f"Selected device for memory requirement {memory_requirement / 1024**2:.1f}MB: {best_device}")
        return best_device
    
    def allocate_devices_for_comparison(self, left_memory_req: int = 0, right_memory_req: int = 0) -> Tuple[str, str]:
        """
        Allocate optimal device combination for dual model comparison
        
        Args:
            left_memory_req: Left model memory requirement
            right_memory_req: Right model memory requirement
            
        Returns:
            (left_device, right_device) device name tuple
        """
        # Get all available CUDA devices
        cuda_devices = [d for d in self._devices.keys() if d.startswith('cuda:') and self._devices[d].is_available]
        
        # If there are multiple CUDA devices, allocate to different devices
        if len(cuda_devices) >= 2:
            # Select the two devices with the most available memory
            cuda_devices.sort(key=lambda d: self._devices[d].available_memory, reverse=True)
            
            left_device = cuda_devices[0]
            right_device = cuda_devices[1]
            
            # Check if memory is sufficient
            if (self._devices[left_device].available_memory >= left_memory_req and 
                self._devices[right_device].available_memory >= right_memory_req):
                print(f"Dual model allocation: left={left_device}, right={right_device}")
                return left_device, right_device
        
        # If only one CUDA device or insufficient memory, check if sharing is possible
        if len(cuda_devices) >= 1:
            cuda_device = cuda_devices[0]
            total_memory_req = left_memory_req + right_memory_req
            
            if self._devices[cuda_device].available_memory >= total_memory_req:
                print(f"Dual model sharing CUDA device: {cuda_device}")
                return cuda_device, cuda_device
        
        # Fallback strategy: one uses CUDA, one uses CPU
        if len(cuda_devices) >= 1:
            cuda_device = cuda_devices[0]
            if self._devices[cuda_device].available_memory >= max(left_memory_req, right_memory_req):
                print(f"Mixed allocation: CUDA={cuda_device}, CPU=cpu")
                return cuda_device, 'cpu'
        
        # Final fallback: both use CPU
        print("Fallback to CPU dual inference")
        return 'cpu', 'cpu'
    
    def get_device_info(self, device_name: str) -> Optional[DeviceInfo]:
        """Get information for the specified device"""
        return self._devices.get(device_name)
    
    def get_all_devices_info(self) -> Dict[str, DeviceInfo]:
        """Get information for all devices"""
        return self._devices.copy()
    
    def estimate_model_memory(self, model_size_mb: float) -> int:
        """
        Estimate memory required for model loading
        
        Args:
            model_size_mb: Model file size (MB)
            
        Returns:
            Estimated memory requirement (bytes)
        """
        # Simple estimation: model file size * 2.5 (considering activation values, gradients, etc.)
        # This is an empirical value, actual may vary by model
        estimated_bytes = int(model_size_mb * 1024 * 1024 * 2.5)
        return estimated_bytes
    
    def clear_cache(self, device_name: Optional[str] = None):
        """Clear device cache"""
        if device_name is None:
            # Clear cache for all CUDA devices
            for device in self._devices:
                if device.startswith('cuda:'):
                    try:
                        device_id = self._devices[device].device_id
                        torch.cuda.set_device(device_id)
                        torch.cuda.empty_cache()
                        print(f"Cleared cache for {device}")
                    except Exception as e:
                        print(f"Error clearing cache for {device}: {e}")
        else:
            if device_name.startswith('cuda:') and device_name in self._devices:
                try:
                    device_id = self._devices[device_name].device_id
                    torch.cuda.set_device(device_id)
                    torch.cuda.empty_cache()
                    print(f"Cleared cache for {device_name}")
                except Exception as e:
                    print(f"Error clearing cache for {device_name}: {e}")
    
    def stop_monitoring(self):
        """Stop device monitoring"""
        self._monitoring_active = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)


# Global device manager instance
device_manager = DeviceManager()