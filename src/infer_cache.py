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
    
    def _get_cache_key(self, ckpt_path: str, device: str, dtype: str) -> str:
        """Generate cache key"""
        return f"{ckpt_path}:{device}:{dtype}"
    
    def get_model_and_meta(self, ckpt_path: str, data_dir: str, device: str, dtype: str, compile_model: bool = False) -> Tuple[Any, Any, Any, Any]:
        """
        Get cached model and metadata, load if not exists
        Smart device allocation version
        """
        # If device not explicitly specified, use device manager to choose the best one
        if device == "auto" or device == DEFAULT_CONFIG["inference"]["device"]:
            # Estimate model memory requirements
            if os.path.exists(ckpt_path):
                model_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
                memory_req = device_manager.estimate_model_memory(model_size_mb)
                device = device_manager.get_best_device(memory_req, prefer_cuda=True)
                print(f"Smart device selection: {device} (model size: {model_size_mb:.1f}MB)")
        
        cache_key = self._get_cache_key(ckpt_path, device, dtype)
        
        with self._lock:
            # Check model cache
            if cache_key in self._cache:
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
            
            # Cache miss or invalid, reload model
            print(f"Loading model: {ckpt_path} -> {device} ({dtype})")
            
            # Clear device cache before loading to free memory
            if device.startswith('cuda:'):
                device_manager.clear_cache(device)
            
            # Set device-related configuration
            device_type = 'cuda' if 'cuda' in device else 'cpu'
            
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location=device)
            model_args = checkpoint['model_args']
            
            # Determine model type
            is_self_attention_model = any(key in model_args for key in [
                'ffn_hidden_mult', 'qkv_bias', 'attn_dropout', 'resid_dropout',
                'ln_eps', 'init_std', 'use_flash_attn', 'pos_encoding_type',
                'rope_base', 'rope_cache_size', 'alibi_bias_scale', 'ffn_activation'
            ])
            
            # Create model
            if is_self_attention_model:
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
                
                gptconf = GPTSelfAttnConfig(**model_args)
                model = GPTSelfAttn(gptconf)
            else:
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
                'gptconf': gptconf
            }
            
            # Load and cache metadata
            meta_key = data_dir
            if meta_key not in self._meta_cache:
                encode, decode = self._load_meta_and_tokenizer(data_dir)
                self._meta_cache[meta_key] = (encode, decode)
            else:
                encode, decode = self._meta_cache[meta_key]
            
            print(f"Model loaded and cached: {cache_key}")
            return model, gptconf, encode, decode
    
    def _load_meta_and_tokenizer(self, data_dir: str) -> Tuple[Any, Any]:
        """Load metadata and tokenizer"""
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"meta.pkl not found at {meta_path}")
        
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        tokenizer_type = meta.get('tokenizer', 'character level')
        stoi, itos = meta['stoi'], meta['itos']
        
        def safe_decode(decode_func, tokens, fallback_char=""):
            try:
                return decode_func(tokens)
            except Exception:
                return fallback_char
        
        if 'old2new' in meta:
            old2new = meta['old2new']
            new2old = {new: old for old, new in old2new.items()}
            
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
        
        return encode, decode
    
    def clear_cache(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._meta_cache.clear()
    
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache information"""
        with self._lock:
            return {
                'model_cache_size': len(self._cache),
                'meta_cache_size': len(self._meta_cache)
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
    compile_model=DEFAULT_CONFIG["inference"]["compile_model"]
):
    """
    Efficient text generation function using cache
    """
    if not prompt.strip():
        yield "Prompt is empty, please provide a starting text."
        return
    
    try:
        # Set random seed
        torch.manual_seed(seed)
        if 'cuda' in device:
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        device_type = DEFAULT_CONFIG["inference"]["device"]
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
    
    except Exception as ex:
        yield f"An unexpected error occurred: {str(ex)}"