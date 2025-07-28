#!/usr/bin/env python3
"""
CUDA Detection Script for Docker Container
Detects if CUDA is available and returns appropriate environment path
"""

import sys
import subprocess
import os

def check_nvidia_smi():
    """Check if nvidia-smi is available and working"""
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def check_cuda_runtime():
    """Check if CUDA runtime libraries are available"""
    try:
        # Check for CUDA runtime library
        result = subprocess.run(['ldconfig', '-p'], 
                              capture_output=True, 
                              text=True, 
                              timeout=5)
        if result.returncode == 0:
            return 'libcudart' in result.stdout
        return False
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def check_cuda_devices():
    """Check if CUDA devices are available"""
    try:
        # Check /proc/driver/nvidia/gpus directory
        if os.path.exists('/proc/driver/nvidia/gpus'):
            gpu_dirs = os.listdir('/proc/driver/nvidia/gpus')
            return len(gpu_dirs) > 0

        # Alternative check: look for nvidia device files
        if os.path.exists('/dev/nvidia0'):
            return True

        # Check for nvidia-ml-py if available
        try:
            result = subprocess.run(['nvidia-smi', '-L'],
                                  capture_output=True,
                                  text=True,
                                  timeout=5)
            if result.returncode == 0 and 'GPU' in result.stdout:
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        return False
    except (OSError, PermissionError):
        return False

def test_torch_cuda():
    """Test if PyTorch can detect CUDA (using CUDA environment)"""
    try:
        # Use CUDA environment to test CUDA availability
        result = subprocess.run([
            '/opt/venv-cuda/bin/python', '-c',
            'import torch; print("CUDA_AVAILABLE:" + str(torch.cuda.is_available())); print("CUDA_COUNT:" + str(torch.cuda.device_count()))'
        ], capture_output=True, text=True, timeout=30)

        if result.returncode == 0:
            output = result.stdout.strip()
            lines = output.split('\n')

            cuda_available = False
            cuda_count = 0

            for line in lines:
                if line.startswith('CUDA_AVAILABLE:'):
                    cuda_available = line.split(':')[1].strip() == 'True'
                elif line.startswith('CUDA_COUNT:'):
                    cuda_count = int(line.split(':')[1].strip())

            return cuda_available and cuda_count > 0
        return False
    except (subprocess.TimeoutExpired, subprocess.SubprocessError, ValueError):
        return False

def main():
    """Main detection logic"""
    print("🔍 Detecting CUDA availability...", file=sys.stderr)

    # Multiple checks for robustness
    checks = {
        "nvidia-smi": check_nvidia_smi(),
        "cuda_runtime": check_cuda_runtime(),
        "cuda_devices": check_cuda_devices(),
        "torch_cuda": test_torch_cuda()
    }

    # Print check results for debugging
    for check_name, result in checks.items():
        status = "✅" if result else "❌"
        print(f"{status} {check_name}: {result}", file=sys.stderr)

    # Determine if CUDA is available
    # We need at least nvidia-smi and torch to detect CUDA
    cuda_available = checks["nvidia-smi"] and checks["torch_cuda"]

    if cuda_available:
        print("🚀 CUDA detected! Using CUDA environment.", file=sys.stderr)
        print("/opt/venv-cuda/bin/python")
    else:
        print("💻 CUDA not available. Using CPU environment.", file=sys.stderr)
        print("/opt/venv-cpu/bin/python")

if __name__ == "__main__":
    main()
