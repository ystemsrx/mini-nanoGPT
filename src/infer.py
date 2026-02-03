# src/infer.py
import os
import pickle
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from src.config import DEFAULT_CONFIG
from src.db_manager import DBManager
from src.gpt_model import GPTConfig, GPT
from src.gpt_self_attn import GPTSelfAttnConfig, GPTSelfAttn


class UnknownTokenError(Exception):
    """Exception raised when encoding encounters tokens not in vocabulary."""
    def __init__(self, unknown_tokens, message=None):
        self.unknown_tokens = unknown_tokens
        if message is None:
            tokens_str = ', '.join(repr(t) for t in unknown_tokens[:10])
            if len(unknown_tokens) > 10:
                tokens_str += f'... (and {len(unknown_tokens) - 10} more)'
            message = f"Prompt contains tokens not in vocabulary: {tokens_str}"
        super().__init__(message)


dbm = DBManager()

def safe_decode(decode_func, tokens, fallback_char=""):
    """
    Safely decodes tokens, handling potential decoding errors.
    """
    try:
        return decode_func(tokens)
    except Exception:
        # Return the fallback character if decoding fails.
        return fallback_char

def generate_text(
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
    Generates text from a single checkpoint and writes inference configuration/history to the database.
    If out_dir ends with .pt, it's treated as the checkpoint path; otherwise, out_dir/ckpt.pt is assumed.
    """
    # Database: ensure model_id & record config
    # DB Integration
    ckpt_dir = out_dir if out_dir.endswith('.pt') else os.path.join(out_dir, 'ckpt.pt')
    model_dir_for_db = os.path.dirname(ckpt_dir)  # Use directory to locate the model
    model_name_for_db = os.path.basename(model_dir_for_db) or "new_model"
    model_id = dbm.get_model_id_by_dir(model_dir_for_db)
    if model_id is None:
        model_id = dbm.register_model(model_name_for_db, model_dir_for_db)

    inf_cfg_dict = dict(
        data_dir=data_dir,
        out_dir=out_dir,
        prompt=prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        device=device,
        dtype=dtype,
        compile_model=compile_model
    )
    dbm.save_inference_config(model_id, inf_cfg_dict)
    # ...

    if not prompt.strip():
        yield "Prompt is empty, please provide a starting text."
        return

    try:
        # Set random seed using Generator for thread-safe deterministic sampling
        # Create a dedicated generator to avoid interference from other threads or code
        rng_generator = torch.Generator(device='cpu')
        rng_generator.manual_seed(seed)
        
        # Also set global seeds for operations that may use global RNG
        torch.manual_seed(seed)
        if 'cuda' in device:
            torch.cuda.manual_seed(seed)
            # Create CUDA generator for GPU operations
            cuda_rng_generator = torch.Generator(device=device)
            cuda_rng_generator.manual_seed(seed)
        else:
            cuda_rng_generator = None
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Extract device type for autocast - always use 'cuda' for any CUDA device
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # Analyze checkpoint path
        ckpt_path = ckpt_dir
        if not os.path.exists(ckpt_path):
            err = f"Error: checkpoint not found at {ckpt_path}."
            yield err
            return

        checkpoint = torch.load(ckpt_path, map_location=device)
        model_args = checkpoint['model_args']
        
        # Get the original training dtype for comparison
        original_dtype = model_args.get('dtype', 'float32')  # Default to float32 if not found
        
        # Determine if this is a self-attention model based on model_args
        # Check for any self-attention specific parameters
        is_self_attention_model = any(key in model_args for key in [
            'ffn_hidden_mult', 'qkv_bias', 'attn_dropout', 'resid_dropout',
            'ln_eps', 'init_std', 'use_flash_attn', 'pos_encoding_type',
            'rope_base', 'rope_cache_size', 'alibi_bias_scale', 'ffn_activation'
        ])
        
        try:
            if is_self_attention_model:
                # Ensure all required parameters have defaults for backward compatibility
                default_self_attn_args = {
                    'ffn_hidden_mult': 4,
                    'qkv_bias': True,
                    'attn_dropout': 0.1,
                    'resid_dropout': 0.1,
                    'ln_eps': 1e-5,
                    'init_std': 0.02,
                    'use_flash_attn': False,
                    'pos_encoding_type': 'rope',
                    'rope_base': 10000,
                    # New optimized parameters with sensible defaults
                    'rope_cache_size': None,
                    'alibi_bias_scale': 1.0,
                    'ffn_activation': 'gelu',
                    'attention_scale_factor': 1.0,
                    'gradient_checkpointing': False
                }
                
                # Merge with saved args, preferring saved values
                for key, default_val in default_self_attn_args.items():
                    if key not in model_args:
                        model_args[key] = default_val
                        print(f"Using default value for {key}: {default_val}")
                
                gptconf = GPTSelfAttnConfig(**model_args)
                model = GPTSelfAttn(gptconf)
            else:
                gptconf = GPTConfig(**model_args)
                model = GPT(gptconf)
            
        except Exception as e:
            err_msg = f"Failed to create model with args {model_args}: {str(e)}"
            yield err_msg
            return

        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        
        # Handle dtype conversion if inference dtype differs from training dtype
        if dtype != original_dtype:
            print(f"Converting model from {original_dtype} to {dtype} for inference")
            try:
                ptdtype_target = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
                model = model.to(dtype=ptdtype_target)
                yield f"Model dtype converted from {original_dtype} to {dtype}\n"
            except Exception as e:
                warning_msg = f"Warning: Failed to convert model dtype from {original_dtype} to {dtype}: {str(e)}. Using original dtype.\n"
                print(warning_msg)
                yield warning_msg
                # Continue with original dtype
                dtype = original_dtype
        
        if compile_model:
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"Warning: Model compilation failed: {str(e)}. Continuing without compilation.")
                # Continue without compilation

        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            yield f"Error: meta.pkl not found at {meta_path}."
            return

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        tokenizer_type = meta.get('tokenizer', 'character level')
        stoi, itos = meta['stoi'], meta['itos']

        # If 'old2new' or 'old2new_mapping' exists in meta, a tokenizer was used.
        # Support both key names for compatibility
        old2new = meta.get('old2new') or meta.get('old2new_mapping')
        if old2new is not None:
            new2old = {new: old for old, new in old2new.items()}

            # Choose encoding/decoding methods based on tokenizer type.
            if tokenizer_type == 'custom_json':
                try:
                    from tokenizers import Tokenizer
                except ImportError as e:
                    raise ImportError(
                        "Model was trained with custom tokenizer, but the `tokenizers` library is not installed. "
                        "Please install it by running: pip install tokenizers"
                    ) from e
                
                # Changed to use the same search path as data_process.py
                tokenizer_path = os.path.join(Path.cwd(), "assets/tokenizer.json")
                if not os.path.exists(tokenizer_path):
                    raise FileNotFoundError(
                        f"Model was trained with custom tokenizer, but tokenizer file not found: {tokenizer_path}. "
                        "Please ensure assets/tokenizer.json exists."
                    )
                
                tokenizer = Tokenizer.from_file(tokenizer_path)

                def encode(s):
                    # Ensure the full prompt is tokenized correctly.
                    ids = tokenizer.encode(s).ids
                    # Check for unknown tokens (not in old2new mapping)
                    unknown_tokens = [id for id in ids if id not in old2new]
                    if unknown_tokens:
                        raise UnknownTokenError(unknown_tokens)
                    return [old2new[id] for id in ids]

                def decode(l):
                    # Use a default value strategy to process all tokens.
                    original_ids = [new2old.get(id, new2old.get(0, 0)) for id in l]
                    return safe_decode(tokenizer.decode, original_ids)

            else:
                # Default method.
                def encode(s):
                    unknown_chars = [ch for ch in s if ch not in stoi]
                    if unknown_chars:
                        raise UnknownTokenError(unknown_chars)
                    return [stoi[ch] for ch in s]
                def decode(l):
                    return ''.join([itos.get(i, '') for i in l])
        else:
            # Character-level encoding (no ID remapping).
            def encode(s):
                unknown_chars = [ch for ch in s if ch not in stoi]
                if unknown_chars:
                    raise UnknownTokenError(unknown_chars)
                return [stoi[ch] for ch in s]
            def decode(l):
                return ''.join([itos.get(i, '') for i in l])

        # Add prompt tokenization validation before generation.
        encoded_prompt = encode(prompt)
        decoded_prompt = decode(encoded_prompt)
        if decoded_prompt != prompt:
            print(f"Warning: Prompt encoding/decoding mismatch. Original: '{prompt}', After encoding/decoding: '{decoded_prompt}'")
            print(f"Number of tokens after encoding: {len(encoded_prompt)}")

        xids = torch.tensor(encoded_prompt, dtype=torch.long, device=device)[None, ...]
        block_size = gptconf.block_size
        if xids.size(1) > block_size:
            yield f"Error: input length ({xids.size(1)}) exceeds block size ({block_size})."
            return

        # Generate text & accumulate output
        accumulated_output = []
        with torch.no_grad():
            with ctx:
                for s_i in range(num_samples):
                    # Reset generator seed for each sample to ensure reproducibility
                    # Each sample uses seed + sample_index for deterministic but different results
                    sample_seed = seed + s_i
                    rng_generator.manual_seed(sample_seed)
                    if cuda_rng_generator is not None:
                        cuda_rng_generator.manual_seed(sample_seed)
                    
                    # Output title at the start of each sample.
                    sample_header = f"Sample {s_i+1}:\n"
                    yield sample_header

                    idx = xids.clone()
                    # First, output the prompt part.
                    current_text = prompt
                    yield current_text
                    
                    # Used to store the complete generated sequence for the current sample.
                    generated_tokens = []
                    last_valid_text = prompt
                    buffer_size = 5  # Buffer size for handling multi-byte characters.
                    
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
                        # Use dedicated generator for thread-safe deterministic sampling
                        if cuda_rng_generator is not None and probs.device.type == 'cuda':
                            idx_next = torch.multinomial(probs, num_samples=1, generator=cuda_rng_generator)
                        else:
                            # For CPU or when CUDA generator not available, use CPU generator
                            probs_cpu = probs.cpu()
                            idx_next_cpu = torch.multinomial(probs_cpu, num_samples=1, generator=rng_generator)
                            idx_next = idx_next_cpu.to(probs.device)
                        idx = torch.cat((idx, idx_next), dim=1)

                        # Add the newly generated token to the list.
                        new_token = idx_next[0].item()
                        generated_tokens.append(new_token)

                        # Attempt to decode the current token sequence.
                        # When using a tokenizer, decode the sequence.
                        if old2new is not None and tokenizer_type == 'custom_json':
                            # Decode the entire sequence.
                            full_tokens = idx[0].tolist()
                            current_text = decode(full_tokens)
                            
                            # Only output the newly added valid part.
                            if len(current_text) > len(last_valid_text):
                                new_text = current_text[len(last_valid_text):]
                                yield new_text
                                last_valid_text = current_text
                        else:
                            # Character-level encoding, each token corresponds to a character, can be decoded directly.
                            current_text = decode(idx[0].tolist())
                            new_text = current_text[len(last_valid_text):]
                            yield new_text
                            last_valid_text = current_text

                    # Sample generation finished, ensure the final content is fully decoded.
                    final_text = decode(idx[0].tolist())
                    if len(final_text) > len(last_valid_text):
                        remaining_text = final_text[len(last_valid_text):]
                        if " " not in remaining_text:
                            yield remaining_text

                    # Save the complete sample.
                    full_sample = f"{sample_header}{final_text}"
                    accumulated_output.append(full_sample)

                    # Add a separator line between samples.
                    if s_i < num_samples - 1:
                        separator = "\n" + "-" * 30 + "\n"
                        yield separator

        # Save plain text inference history for standalone CLI usage
        # Note: When used via UI, the UI layer saves HTML-formatted history separately
        final_text = "\n\n".join(accumulated_output)
        dbm.save_inference_history(model_id, final_text)

    except Exception as ex:
        yield f"An unexpected error occurred: {str(ex)}"