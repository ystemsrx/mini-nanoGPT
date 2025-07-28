# src/infer_cache.py
import os
import pickle
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Tuple, Any

import torch
import torch.nn.functional as F

from src.config import DEFAULT_CONFIG
from src.gpt_model import GPTConfig, GPT
from src.gpt_self_attn import GPTSelfAttnConfig, GPTSelfAttn
from src.device_manager import device_manager
from src.db_manager import DBManager


class ModelCache:
    """
    Efficient model caching system with model reuse and memory management
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
            self._cache: Dict[str, Any] = {}
            self._meta_cache: Dict[str, Any] = {}
            self._lock = threading.RLock()
            self._initialized = True
    
    def _get_cache_key(self, ckpt_path: str, device: str, dtype: str, model_type: str = "unknown") -> str:
        """Generate cache key including model type to avoid conflicts"""
        return f"{ckpt_path}:{device}:{dtype}:{model_type}"
    
    def _detect_model_type(self, model_args: dict) -> str:
        """
        Robustly detect model type from model_args
        Returns 'self_attention' or 'basic'
        """
        # Check for self-attention specific parameters
        self_attn_keys = [
            'ffn_hidden_mult', 'qkv_bias', 'attn_dropout', 'resid_dropout',
            'ln_eps', 'init_std', 'use_flash_attn', 'pos_encoding_type',
            'rope_base', 'rope_cache_size', 'alibi_bias_scale', 'ffn_activation',
            'attention_scale_factor', 'gradient_checkpointing'
        ]
        
        # Also check for explicit use_self_attention flag
        if model_args.get('use_self_attention', False):
            return 'self_attention'
        
        # Check if any self-attention specific parameters exist
        has_self_attn_params = any(key in model_args for key in self_attn_keys)
        
        return 'self_attention' if has_self_attn_params else 'basic'
    
    def get_model_and_meta(self, ckpt_path: str, data_dir: str, device: str, dtype: str, compile_model: bool = False) -> Tuple[Any, Any, Any, Any]:
        """
        Get cached model and metadata, load if not exists
        Smart device allocation version with improved model type detection
        """
        # If device not explicitly specified, use device manager to choose the best one
        if device == "auto" or device == DEFAULT_CONFIG["inference"]["device"]:
            # Estimate model memory requirements
            if os.path.exists(ckpt_path):
                model_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                memory_req = device_manager.estimate_model_memory(model_size_mb)
                device = device_manager.get_best_device(memory_req, prefer_cuda=True)
                print(f"Smart device selection: {device} (model size: {model_size_mb:.1f}MB)")
        
        # Pre-load checkpoint to detect model type for cache key
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu')  # Load to CPU first to check args
            model_args = checkpoint['model_args']
            model_type = self._detect_model_type(model_args)
            print(f"Detected model type: {model_type} for {os.path.basename(ckpt_path)}")
        except Exception as e:
            print(f"Warning: Failed to load checkpoint for type detection: {e}")
            model_type = "unknown"
            checkpoint = None
            model_args = {}
        
        cache_key = self._get_cache_key(ckpt_path, device, dtype, model_type)
        
        with self._lock:
            # Check if checkpoint file has been modified since last cache
            should_reload = False
            if os.path.exists(ckpt_path):
                current_mtime = os.path.getmtime(ckpt_path)
                
                # Check if we have cached modification time
                mtime_key = f"{cache_key}_mtime"
                if mtime_key in self._cache:
                    cached_mtime = self._cache[mtime_key]
                    if current_mtime > cached_mtime:
                        print(f"Checkpoint file modified, clearing cache for: {ckpt_path}")
                        should_reload = True
                        # Remove old cache entries for this checkpoint
                        keys_to_remove = [k for k in self._cache.keys() if k.startswith(ckpt_path)]
                        for k in keys_to_remove:
                            del self._cache[k]
                else:
                    # First time caching this file
                    should_reload = True
                
                # Store current modification time
                self._cache[mtime_key] = current_mtime
            
            # Check model cache
            if cache_key in self._cache and not should_reload:
                cached_data = self._cache[cache_key]
                model = cached_data['model']
                gptconf = cached_data['gptconf']
                
                # Verify if model is still on the correct device
                if hasattr(model, 'parameters'):
                    try:
                        first_param = next(model.parameters())
                        if str(first_param.device) != device:
                            # Device mismatch, need to reload
                            print(f"Device mismatch (expected: {device}, actual: {first_param.device}), reloading model")
                            del self._cache[cache_key]
                        else:
                            # Cache hit, get metadata
                            meta_key = data_dir
                            if meta_key in self._meta_cache:
                                encode, decode = self._meta_cache[meta_key]
                                print(f"Cache hit: {cache_key}")
                                return model, gptconf, encode, decode
                    except StopIteration:
                        # Model has no parameters, remove from cache
                        del self._cache[cache_key]
                    except Exception as e:
                        print(f"Warning: Cache validation error: {e}, removing from cache")
                        del self._cache[cache_key]
            
            # Also check and clean old format cache entries that might conflict
            old_cache_key = f"{ckpt_path}:{device}:{dtype}"
            if old_cache_key in self._cache:
                print(f"Removing old format cache entry: {old_cache_key}")
                del self._cache[old_cache_key]
            
            # Cache miss or invalid, reload model
            print(f"Loading {model_type} model: {ckpt_path} -> {device} ({dtype})")
            
            # Clear device cache before loading to free memory
            if device.startswith('cuda:'):
                device_manager.clear_cache(device)
            
            # Set device-related configuration
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            
            # Load checkpoint if not already loaded
            if checkpoint is None:
                checkpoint = torch.load(ckpt_path, map_location=device)
                model_args = checkpoint['model_args']
                model_type = self._detect_model_type(model_args)
            
            # Create model based on detected type
            try:
                if model_type == 'self_attention':
                    # Get default parameters from config file instead of hardcoding
                    training_config = DEFAULT_CONFIG["training"]
                    default_self_attn_args = {
                        'ffn_hidden_mult': training_config["ffn_hidden_mult"],
                        'qkv_bias': training_config["qkv_bias"],
                        'attn_dropout': training_config["attn_dropout"],
                        'resid_dropout': training_config["resid_dropout"],
                        'ln_eps': training_config["ln_eps"],
                        'init_std': training_config["init_std"],
                        'use_flash_attn': training_config["use_flash_attn"],
                        'pos_encoding_type': training_config["pos_encoding_type"],
                        'rope_base': training_config["rope_base"],
                        'rope_cache_size': training_config["rope_cache_size"],
                        'alibi_bias_scale': training_config["alibi_bias_scale"],
                        'ffn_activation': training_config["ffn_activation"],
                        'attention_scale_factor': training_config["attention_scale_factor"],
                        'gradient_checkpointing': training_config["gradient_checkpointing"]
                    }
                    
                    for key, default_val in default_self_attn_args.items():
                        if key not in model_args:
                            model_args[key] = default_val
                    
                    print(f"Creating GPTSelfAttn model with config: {list(model_args.keys())}")
                    gptconf = GPTSelfAttnConfig(**model_args)
                    model = GPTSelfAttn(gptconf)
                else:
                    print(f"Creating basic GPT model with config: {list(model_args.keys())}")
                    gptconf = GPTConfig(**model_args)
                    model = GPT(gptconf)
                
                # Load state dict
                state_dict = checkpoint['model']
                unwanted_prefix = '_orig_mod.'
                for k, v in list(state_dict.items()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                
                model.load_state_dict(state_dict)
                model.eval()
                model.to(device)
                
                # Handle data type conversion
                original_dtype = model_args.get('dtype', 'float32')
                if dtype != original_dtype:
                    try:
                        ptdtype_target = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
                        model = model.to(dtype=ptdtype_target)
                        print(f"Data type conversion: {original_dtype} -> {dtype}")
                    except Exception as e:
                        print(f"Warning: Data type conversion failed: {e}")
                        dtype = original_dtype
                
                # Compile model (if needed)
                if compile_model:
                    try:
                        model = torch.compile(model)
                        print("Model compilation completed")
                    except Exception as e:
                        print(f"Warning: Model compilation failed: {e}")
                
                # Cache model
                self._cache[cache_key] = {
                    'model': model,
                    'gptconf': gptconf,
                    'model_type': model_type
                }
                
                print(f"✅ Successfully loaded and cached {model_type} model")
                
            except Exception as e:
                print(f"❌ Failed to create {model_type} model: {e}")
                import traceback
                print(traceback.format_exc())
                raise RuntimeError(f"Model creation failed for {model_type}: {e}")
            
            # Load and cache metadata
            meta_key = data_dir
            if meta_key not in self._meta_cache:
                encode, decode = self._load_meta_and_tokenizer(data_dir)
                self._meta_cache[meta_key] = (encode, decode)
            else:
                encode, decode = self._meta_cache[meta_key]
            
            return model, gptconf, encode, decode

    def _load_meta_and_tokenizer(self, data_dir: str) -> Tuple[Any, Any]:
        """Load metadata and tokenizer"""
        meta_path = os.path.join(data_dir, 'meta.pkl')
        stoi, itos = {}, {}
        
        def safe_decode(decode_func, tokens, fallback_char=""):
            try:
                return decode_func(tokens)
            except:
                # Try token by token
                result = ""
                for token in tokens:
                    try:
                        result += decode_func([token])
                    except:
                        result += fallback_char
                return result
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                    stoi = meta.get('stoi', {})
                    itos = meta.get('itos', {})
                    
                    # Handle different token mapping formats
                    if 'old2new' in meta and 'new2old' in meta:
                        old2new = meta['old2new']
                        new2old = meta['new2old']
                        tokenizer_type = meta.get('tokenizer_type', 'char_level')
                        
                        if tokenizer_type == 'custom_json':
                            try:
                                from tokenizers import Tokenizer
                                tokenizer_path = os.path.join(Path.cwd(), "assets/tokenizer.json")
                                if os.path.exists(tokenizer_path):
                                    tokenizer = Tokenizer.from_file(tokenizer_path)
                                    
                                    def encode(s):
                                        ids = tokenizer.encode(s).ids
                                        return [old2new.get(id, old2new.get(0, 0)) for id in ids]
                                    
                                    def decode(l):
                                        original_ids = [new2old.get(id, new2old.get(0, 0)) for id in l]
                                        return safe_decode(tokenizer.decode, original_ids)
                                else:
                                    def encode(s):
                                        return [stoi.get(ch, 0) for ch in s]
                                    def decode(l):
                                        return ''.join([itos.get(i, '') for i in l])
                            except ImportError:
                                def encode(s):
                                    return [stoi.get(ch, 0) for ch in s]
                                def decode(l):
                                    return ''.join([itos.get(i, '') for i in l])
                        
                        elif tokenizer_type == 'gpt2':
                            import tiktoken
                            enc = tiktoken.get_encoding("gpt2")
                            
                            def encode(s):
                                ids = enc.encode(s, allowed_special={"<|endoftext|>"})
                                return [old2new.get(id, old2new.get(0, 0)) for id in ids]
                            
                            def decode(l):
                                original_ids = [new2old.get(id, new2old.get(0, 0)) for id in l]
                                return safe_decode(enc.decode, original_ids)
                        
                        else:
                            def encode(s):
                                return [stoi.get(ch, 0) for ch in s]
                            def decode(l):
                                return ''.join([itos.get(i, '') for i in l])
                    else:
                        def encode(s):
                            return [stoi.get(ch, 0) for ch in s]
                        def decode(l):
                            return ''.join([itos.get(i, '') for i in l])
            except:
                def encode(s):
                    return [stoi.get(ch, 0) for ch in s]
                def decode(l):
                    return ''.join([itos.get(i, '') for i in l])
        else:
            def encode(s):
                return [stoi.get(ch, 0) for ch in s]
            def decode(l):
                return ''.join([itos.get(i, '') for i in l])
        
        return encode, decode
    
    def clear_cache(self):
        """Force complete cache cleanup with support for new cache key format"""
        with self._lock:
            print("🧹 Starting complete cache cleanup...")
            
            # Clean up all model caches by moving them to CPU first
            for cache_key, cached_data in list(self._cache.items()):
                if isinstance(cached_data, dict) and 'model' in cached_data:
                    try:
                        model = cached_data['model']
                        model_type = cached_data.get('model_type', 'unknown')
                        if hasattr(model, 'cpu'):
                            model.cpu()
                        del model
                    except Exception as e:
                        print(f"Warning during model cleanup: {e}")
                elif not cache_key.endswith('_mtime'):
                    # Clean up any other cache entries that are not modification time stamps
                    pass
            
            # Clear all cache dictionaries
            self._cache.clear()
            self._meta_cache.clear()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            print("✅ Cache completely cleared!")
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache information"""
        with self._lock:
            return {
                'model_cache_size': len(self._cache),
                'meta_cache_size': len(self._meta_cache)
            }
    
    def debug_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information for debugging"""
        with self._lock:
            cache_details = {}
            model_count = 0
            
            for cache_key, cached_data in self._cache.items():
                if isinstance(cached_data, dict) and 'model' in cached_data:
                    model_count += 1
                    model_type = cached_data.get('model_type', 'unknown')
                    cache_details[cache_key] = {
                        'model_type': model_type,
                        'has_model': True
                    }
                elif cache_key.endswith('_mtime'):
                    cache_details[cache_key] = {
                        'is_mtime': True,
                        'value': cached_data
                    }
                else:
                    cache_details[cache_key] = {
                        'type': str(type(cached_data)),
                        'has_model': False
                    }
            
            return {
                'total_cache_entries': len(self._cache),
                'model_count': model_count,
                'meta_cache_size': len(self._meta_cache),
                'cache_details': cache_details,
                'meta_cache_keys': list(self._meta_cache.keys())
            }


def cached_generate_text(
    data_dir,
    out_dir,
    prompt=DEFAULT_CONFIG["inference"]["prompt"],
    num_samples=DEFAULT_CONFIG["inference"]["num_samples"],
    max_new_tokens=DEFAULT_CONFIG["inference"]["max_new_tokens"],
    temperature=DEFAULT_CONFIG["inference"]["temperature"],
    top_k=DEFAULT_CONFIG["inference"]["top_k"],
    seed=DEFAULT_CONFIG["inference"]["seed"],
    device=DEFAULT_CONFIG["inference"]["device"],
    dtype=DEFAULT_CONFIG["inference"]["dtype"],
    compile_model=DEFAULT_CONFIG["inference"]["compile_model"],
    auto_clear_cache=True  # New parameter to control automatic cache cleanup
):
    """
    Efficient text generation function using cache
    """
    if not prompt.strip():
        yield "Prompt is empty, please provide a starting text."
        return
    
    cache = None
    try:
        # Database integration - get model_id for saving inference history
        dbm = DBManager()
        ckpt_dir = out_dir if out_dir.endswith('.pt') else os.path.join(out_dir, 'ckpt.pt')
        model_dir_for_db = os.path.dirname(ckpt_dir)
        model_name_for_db = os.path.basename(model_dir_for_db) or "new_model"
        model_id = dbm.get_model_id_by_dir(model_dir_for_db)
        if model_id is None:
            model_id = dbm.register_model(model_name_for_db, model_dir_for_db)

        # Set random seed
        torch.manual_seed(seed)
        if 'cuda' in device:
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        # Extract device type for autocast - always use 'cuda' for any CUDA device
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        
        # Prepare checkpoint path
        ckpt_path = out_dir if out_dir.endswith('.pt') else os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            yield f"Error: checkpoint not found at {ckpt_path}."
            return
        
        # Use cache to get model and metadata
        cache = ModelCache()
        model, gptconf, encode, decode = cache.get_model_and_meta(ckpt_path, data_dir, device, dtype, compile_model)
        
        # Validate prompt encoding
        encoded_prompt = encode(prompt)
        decoded_prompt = decode(encoded_prompt)
        if decoded_prompt != prompt:
            print(f"Warning: Prompt encoding/decoding mismatch.")
        
        xids = torch.tensor(encoded_prompt, dtype=torch.long, device=device)[None, ...]
        block_size = gptconf.block_size
        if xids.size(1) > block_size:
            yield f"Error: input length ({xids.size(1)}) exceeds block size ({block_size})."
            return
        
        # Generate text
        accumulated_output = []
        with torch.no_grad():
            with ctx:
                for s_i in range(num_samples):
                    sample_header = f"Sample {s_i+1}:\n"
                    yield sample_header
                    
                    idx = xids.clone()
                    current_text = prompt
                    yield current_text
                    
                    last_valid_text = prompt
                    
                    for token_i in range(max_new_tokens):
                        if idx.size(1) == 0:
                            yield "Can't generate an empty sequence."
                            return
                        
                        idx_cond = idx[:, -block_size:]
                        logits, _ = model(idx_cond)
                        logits = logits[:, -1, :] / temperature
                        
                        if top_k is not None and top_k > 0:
                            v, _ = torch.topk(logits, top_k)
                            top_value = v[:, -1].unsqueeze(-1)
                            logits[logits < top_value] = -float('Inf')
                        
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)
                        
                        # Decode new text
                        current_text = decode(idx[0].tolist())
                        if len(current_text) > len(last_valid_text):
                            new_text = current_text[len(last_valid_text):]
                            yield new_text
                            last_valid_text = current_text
                    
                    # Complete current sample
                    final_text = decode(idx[0].tolist())
                    if len(final_text) > len(last_valid_text):
                        remaining_text = final_text[len(last_valid_text):]
                        if remaining_text.strip():
                            yield remaining_text
                    
                    full_sample = f"{sample_header}{final_text}"
                    accumulated_output.append(full_sample)
                    
                    if s_i < num_samples - 1:
                        separator = "\n" + "-" * 30 + "\n"
                        yield separator

        # Save inference history to database
        final_text = "\n\n".join(accumulated_output)
        dbm.save_inference_history(model_id, final_text)
        
        # Decide whether to auto-clear cache based on parameter
        if auto_clear_cache and cache:
            cache.clear_cache()
            print("✅ Inference completed, cache auto-cleared")
        else:
            print("ℹ️ Inference completed, cache retained for reuse")
    
    except Exception as ex:
        yield f"An unexpected error occurred: {str(ex)}"
        # Decide whether to clear cache on error based on auto_clear_cache parameter
        if auto_clear_cache:
            try:
                if cache is None:
                    cache = ModelCache()
                cache.clear_cache()
                print("❌ Inference error, cache auto-cleared")
            except Exception as cleanup_error:
                print(f"Warning: Error cache cleanup failed: {cleanup_error}")
        else:
            print("⚠️ Inference error occurred, cache retained")