# src/sft.py
"""
Supervised Fine-Tuning (SFT) module for mini-nanoGPT.
Supports Alpaca-format datasets and Qwen chat template.
"""

import os
import json
import math
from pathlib import Path
from contextlib import nullcontext
from typing import List, Dict, Optional, Tuple, Generator

import numpy as np
import torch
import torch.nn.functional as F

from src.config import DEFAULT_CONFIG
from src.db_manager import DBManager
from src.gpt_model import GPTConfig, GPT, configure_optimizers
from src.gpt_self_attn import GPTSelfAttnConfig, GPTSelfAttn, configure_optimizers_self_attn

dbm = DBManager()

# Special token IDs from Qwen tokenizer.json
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "<|im_end|>"
IM_START_ID = 151644
IM_END_ID = 151645
EOT_ID = 151643

# Global stop signal for SFT training
sft_stop_signal = False


def sanitize_special_tokens(text: str) -> str:
    """
    Sanitize text to prevent literal special tokens from causing boundary detection issues.
    Replaces literal special token strings with escaped/safe versions.
    
    This is important because if user input contains literal '<|im_start|>' or '<|im_end|>',
    the tokenizer will convert them to special token IDs, which can:
    1. Cause incorrect boundary detection in tokenize_sft_sample
    2. Allow prompt injection attacks
    
    We replace them with visually similar but safe alternatives.
    Using fullwidth vertical bars (U+FF5C) which are less likely to be NFKC-normalized back.
    """
    # Replace literal special tokens with safe Unicode alternatives
    # Using fullwidth vertical bar ｜ (U+FF5C) instead of regular | to prevent NFKC normalization
    replacements = [
        (IM_START_TOKEN, "<｜im_start｜>"),  # Using fullwidth vertical bar ｜
        (IM_END_TOKEN, "<｜im_end｜>"),
        ("<|endoftext|>", "<｜endoftext｜>"),
    ]
    
    result = text
    for original, replacement in replacements:
        result = result.replace(original, replacement)
    
    return result


def stop_sft_training():
    """Sets a global stop signal to stop SFT training."""
    global sft_stop_signal
    sft_stop_signal = True


def validate_alpaca_format(data: List[Dict]) -> Tuple[bool, str]:
    """
    Validates that the dataset is in Alpaca format.
    
    Alpaca format requires:
    - Each item must have 'instruction' field (required)
    - 'input' field (optional, can be empty)
    - 'output' field (required)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(data, list):
        return False, "Dataset must be a list of dictionaries"
    
    if len(data) == 0:
        return False, "Dataset is empty"
    
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"Item {i} is not a dictionary"
        
        if 'instruction' not in item:
            return False, f"Item {i} missing required field: 'instruction'"
        
        if 'output' not in item:
            return False, f"Item {i} missing required field: 'output'"
        
        if not isinstance(item.get('instruction', ''), str):
            return False, f"Item {i}: 'instruction' must be a string"
        
        if not isinstance(item.get('output', ''), str):
            return False, f"Item {i}: 'output' must be a string"
        
        # 'input' is optional but must be string if present
        if 'input' in item and not isinstance(item['input'], str):
            return False, f"Item {i}: 'input' must be a string"
    
    return True, f"Valid Alpaca format with {len(data)} samples"


def load_sft_dataset(file_path: str = None, dir_path: str = None) -> Tuple[List[Dict], str]:
    """
    Load SFT dataset from JSON file or directory of JSON files.
    
    Returns:
        Tuple of (dataset, status_message)
    """
    all_data = []
    
    if file_path and os.path.isfile(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                all_data.extend(data)
            else:
                return [], f"Error: File does not contain a list"
        except json.JSONDecodeError as e:
            return [], f"JSON parse error: {str(e)}"
        except Exception as e:
            return [], f"Error loading file: {str(e)}"
    
    elif dir_path and os.path.isdir(dir_path):
        json_files = list(Path(dir_path).glob("*.json"))
        if not json_files:
            return [], f"No JSON files found in {dir_path}"
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
            except Exception as e:
                print(f"Warning: Could not load {json_file}: {e}")
    else:
        return [], "No valid file or directory specified"
    
    # Validate format
    is_valid, msg = validate_alpaca_format(all_data)
    if not is_valid:
        return [], msg
    
    return all_data, msg


def apply_chat_template(
    instruction: str,
    input_text: str = "",
    output: str = "",
    system_prompt: str = "You are a helpful assistant."
) -> str:
    """
    Apply Qwen chat template to an Alpaca sample.
    
    Template format:
    <|im_start|>system
    {system}<|im_end|>
    <|im_start|>user
    {instruction} {input}<|im_end|>
    <|im_start|>assistant
    {output}<|im_end|>
    """
    result = ""
    
    # System message
    result += f"{IM_START_TOKEN}system\n{system_prompt}{IM_END_TOKEN}\n"
    
    # User message (combine instruction and input)
    user_content = instruction
    if input_text and input_text.strip():
        user_content += f"\n{input_text}"
    result += f"{IM_START_TOKEN}user\n{user_content}{IM_END_TOKEN}\n"
    
    # Assistant message
    result += f"{IM_START_TOKEN}assistant\n{output}{IM_END_TOKEN}"
    
    return result


def format_chat_for_inference(
    messages: List[Dict[str, str]],
    system_prompt: str = "You are a helpful assistant."
) -> str:
    """
    Format conversation history for inference with Qwen template.
    
    Args:
        messages: List of {"role": "user"|"assistant", "content": str}
        system_prompt: System message
    
    Returns:
        Formatted string ready for tokenization
    """
    result = f"{IM_START_TOKEN}system\n{system_prompt}{IM_END_TOKEN}\n"
    
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Sanitize user/assistant content to prevent prompt injection
        content = sanitize_special_tokens(content)
        result += f"{IM_START_TOKEN}{role}\n{content}{IM_END_TOKEN}\n"
    
    # End with assistant prompt to generate response
    result += f"{IM_START_TOKEN}assistant\n"
    
    return result


def tokenize_sft_sample(
    sample: Dict,
    tokenizer,
    max_seq_length: int,
    system_prompt: str = "You are a helpful assistant.",
    vocab_size: int = None,
    old2new_mapping: Dict[int, int] = None,
    return_error: bool = False
) -> Optional[Tuple[List[int], List[int]]]:
    """
    Tokenize a single SFT sample with chat template.
    
    For SFT training, we only compute loss on the assistant's response.
    The prompt (system + user) is masked with -100 in labels.
    
    Training format:
        <|im_start|>system\n{system}<|im_end|>\n
        <|im_start|>user\n{instruction} {input}<|im_end|>\n
        <|im_start|>assistant\n{output}<|im_end|>
                              ^--- only compute loss from here
    
    Note: We find the assistant response boundary by locating IM_START token 
    followed by "assistant" in the tokenized sequence, rather than tokenizing
    prefix separately (which can cause token boundary misalignment).
    
    Args:
        sample: Alpaca-format sample dict
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        system_prompt: System prompt for chat template
        vocab_size: Model vocabulary size for validation
        old2new_mapping: Mapping from original token IDs to remapped IDs (from data processing)
        return_error: If True, return (None, error_reason) instead of None on failure
    
    Returns:
        Tuple of (input_ids, labels) or None if sequence is too long or has invalid tokens.
        If return_error=True, returns (None, error_reason_string) on failure.
    """
    instruction = sample.get('instruction', '')
    input_text = sample.get('input', '')
    output = sample.get('output', '')
    
    # C2) Sanitize input to prevent literal special tokens from causing issues
    # This prevents prompt injection and boundary detection errors
    instruction = sanitize_special_tokens(instruction)
    input_text = sanitize_special_tokens(input_text)
    output = sanitize_special_tokens(output)
    # Note: system_prompt is trusted and not sanitized
    
    # Apply full chat template (with assistant's response)
    full_text = apply_chat_template(instruction, input_text, output, system_prompt)
    
    # Tokenize full text (this gives original token IDs from the tokenizer)
    original_tokens = tokenizer.encode(full_text).ids
    
    # C1) Find the boundary where assistant's response starts
    # Use complete marker tokenization for more reliable boundary detection
    # This avoids token boundary misalignment issues from separate tokenization
    assistant_header = f"{IM_START_TOKEN}assistant\n"
    assistant_header_tokens = tokenizer.encode(assistant_header).ids
    
    # Find all occurrences of the complete assistant header pattern
    # For single-turn SFT samples, there should be exactly one match
    header_len = len(assistant_header_tokens)
    matches = []
    
    for i in range(len(original_tokens) - header_len + 1):
        # Check if the sequence starting at i matches the assistant header
        if original_tokens[i:i + header_len] == assistant_header_tokens:
            matches.append(i + header_len)  # Position after the header
    
    if len(matches) == 0:
        # No match found - skip this sample
        if return_error:
            return (None, "no_assistant_header")
        return None
    
    if len(matches) > 1:
        # Multiple matches found - ambiguous boundary, skip this sample
        # This can happen if user input contains literal special tokens that weren't sanitized
        # or if the sample incorrectly has multiple assistant turns
        if return_error:
            return (None, "multiple_assistant_headers")
        return None
    
    prompt_length = matches[0]
    
    # Truncate if sequence is too long (instead of skipping)
    if len(original_tokens) > max_seq_length:
        original_tokens = original_tokens[:max_seq_length]
        # If prompt itself exceeds max_seq_length, skip this sample (no assistant content left)
        if prompt_length >= max_seq_length:
            if return_error:
                return (None, f"prompt_too_long:{prompt_length}>{max_seq_length}")
            return None
    
    # Remap token IDs if mapping is provided (model was trained with remapped IDs)
    if old2new_mapping is not None:
        tokens = []
        for orig_id in original_tokens:
            if orig_id not in old2new_mapping:
                # Token not in vocabulary - skip this sample
                if return_error:
                    return (None, f"token_not_in_vocab:{orig_id}")
                return None
            tokens.append(old2new_mapping[orig_id])
    else:
        tokens = original_tokens
    
    # Validate token IDs are within vocab range
    if vocab_size is not None:
        max_token_id = max(tokens) if tokens else 0
        if max_token_id >= vocab_size:
            if return_error:
                return (None, f"token_out_of_range:{max_token_id}>={vocab_size}")
            return None  # Skip samples with out-of-range tokens
    
    # Create labels: mask prompt tokens with -100, only compute loss on assistant's response
    # The model learns to predict: {output}<|im_end|>
    # Given the prompt: <|im_start|>system...user...<|im_start|>assistant\n
    input_ids = tokens
    labels = [-100] * prompt_length + tokens[prompt_length:]
    
    return input_ids, labels


def prepare_sft_batch(
    samples: List[Dict],
    tokenizer,
    max_seq_length: int,
    system_prompt: str = "You are a helpful assistant.",
    vocab_size: int = None,
    old2new_mapping: Dict[int, int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare a batch of SFT samples.
    
    Args:
        samples: List of Alpaca-format sample dicts
        tokenizer: HuggingFace tokenizer
        max_seq_length: Maximum sequence length
        system_prompt: System prompt for chat template
        vocab_size: Model vocabulary size for validation
        old2new_mapping: Mapping from original token IDs to remapped IDs
    
    Returns:
        Tuple of (input_ids, labels) tensors
    """
    batch_input_ids = []
    batch_labels = []
    
    for sample in samples:
        result = tokenize_sft_sample(sample, tokenizer, max_seq_length, system_prompt, vocab_size, old2new_mapping)
        if result is not None:
            input_ids, labels = result
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
    
    if not batch_input_ids:
        return None, None
    
    # Pad sequences (use the remapped pad token ID, which is 0 after remapping)
    max_len = max(len(ids) for ids in batch_input_ids)
    padded_input_ids = []
    padded_labels = []
    
    for input_ids, labels in zip(batch_input_ids, batch_labels):
        pad_len = max_len - len(input_ids)
        padded_input_ids.append(input_ids + [0] * pad_len)
        padded_labels.append(labels + [-100] * pad_len)  # -100 is ignored in loss
    
    return (
        torch.tensor(padded_input_ids, dtype=torch.long),
        torch.tensor(padded_labels, dtype=torch.long)
    )


def sft_train_generator(
    base_model_ckpt_path: str,
    data_dir: str,
    dataset: List[Dict],
    out_dir: str,
    epochs: int = DEFAULT_CONFIG["sft"]["epochs"],
    learning_rate: float = DEFAULT_CONFIG["sft"]["learning_rate"],
    batch_size: int = DEFAULT_CONFIG["sft"]["batch_size"],
    max_seq_length: int = DEFAULT_CONFIG["sft"]["max_seq_length"],
    gradient_accumulation_steps: int = DEFAULT_CONFIG["sft"]["gradient_accumulation_steps"],
    warmup_ratio: float = DEFAULT_CONFIG["sft"]["warmup_ratio"],
    system_prompt: str = DEFAULT_CONFIG["sft"]["system_prompt"],
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    dtype: str = "float16"
) -> Generator:
    """
    Generator-based SFT training loop.
    
    The model was trained with remapped token IDs (consecutive integers starting from 0),
    so we need to load the old2new_mapping from meta.pkl to convert tokenizer output
    to the correct token IDs used by the model.
    
    Yields:
        Tuple of (progress_html, log_message, plot_data)
    """
    global sft_stop_signal
    sft_stop_signal = False
    
    empty_plot_data = ([], [], [], [])
    
    def make_progress_html(progress_val, max_val, color='blue'):
        return (
            f"<div style='width: 100%; height: 20px; margin-bottom: 5px;'>"
            f"<progress value='{progress_val}' max='{max_val if max_val > 0 else 1}' "
            f"style='width: 100%; height: 20px; color: {color};'></progress>"
            "</div>"
        )
    
    try:
        # Load tokenizer
        tokenizer_path = Path.cwd() / "assets" / "tokenizer.json"
        if not tokenizer_path.exists():
            yield (make_progress_html(0, 1, 'red'), 
                   f"Error: tokenizer.json not found at {tokenizer_path}", 
                   empty_plot_data)
            return
        
        from tokenizers import Tokenizer
        import pickle
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # Load meta.pkl to get old2new_mapping (token ID remapping from data processing)
        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            yield (make_progress_html(0, 1, 'red'),
                   f"Error: meta.pkl not found at {meta_path}. Please ensure data processing was completed.",
                   empty_plot_data)
            return
        
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        old2new_mapping = meta.get('old2new_mapping', None)
        if old2new_mapping is None:
            yield (make_progress_html(0, 1, 'red'),
                   f"Error: old2new_mapping not found in meta.pkl. The model may have been trained with an older version. "
                   f"Please re-process the training data with the current version.",
                   empty_plot_data)
            return
        
        # Check if special tokens are in the mapping (required for SFT)
        missing_special_tokens = []
        for token_name, token_id in [("IM_START", IM_START_ID), ("IM_END", IM_END_ID), ("EOT", EOT_ID)]:
            if token_id not in old2new_mapping:
                missing_special_tokens.append(f"{token_name}({token_id})")
        
        if missing_special_tokens:
            yield (make_progress_html(0, 1, 'red'),
                   f"Error: Special tokens required for SFT are missing from vocabulary: {', '.join(missing_special_tokens)}. "
                   f"Please re-process the training data with 'Use GPT2 Tokenizer' option (which uses the custom Qwen tokenizer). "
                   f"The data processing will now automatically include these special tokens.",
                   empty_plot_data)
            return
        
        # Check chat role tokens (system, user, assistant)
        # Token IDs: system=8948, user=872, assistant=77091
        CHAT_ROLE_TOKENS = [("system", 8948), ("user", 872), ("assistant", 77091)]
        missing_role_tokens = []
        for token_name, token_id in CHAT_ROLE_TOKENS:
            if token_id not in old2new_mapping:
                missing_role_tokens.append(f"'{token_name}'({token_id})")
        
        if missing_role_tokens:
            yield (make_progress_html(0, 1, 'red'),
                   f"<div style='padding: 10px; background: #fff3f3; border-radius: 8px; border: 1px solid #ffcccc;'>"
                   f"<b style='color: #cc0000;'>❌ Error: Chat role tokens missing from vocabulary</b><br><br>"
                   f"Missing tokens: <code>{', '.join(missing_role_tokens)}</code><br><br>"
                   f"<b>Why this matters:</b><br>"
                   f"The SFT chat template uses 'system', 'user', 'assistant' as role markers. "
                   f"If these tokens are not in the model's vocabulary, SFT cannot work.<br><br>"
                   f"<div style='background: #e8f4fd; padding: 8px; border-radius: 4px;'>"
                   f"<b>💡 Solution:</b> Please <b>re-process the training data</b> in the Data Processing tab. "
                   f"The latest version automatically includes these chat role tokens."
                   f"</div></div>",
                   empty_plot_data)
            return
        
        model_vocab_size = meta.get('vocab_size', 0)
        
        # Load base model checkpoint
        if not os.path.exists(base_model_ckpt_path):
            yield (make_progress_html(0, 1, 'red'),
                   f"Error: Base model checkpoint not found: {base_model_ckpt_path}",
                   empty_plot_data)
            return
        
        checkpoint = torch.load(base_model_ckpt_path, map_location=device)
        model_args = checkpoint['model_args']
        
        # Determine model type
        is_self_attention_model = any(key in model_args for key in [
            'ffn_hidden_mult', 'qkv_bias', 'attn_dropout', 'resid_dropout'
        ])
        
        if is_self_attention_model:
            gptconf = GPTSelfAttnConfig(**model_args)
            model = GPTSelfAttn(gptconf)
        else:
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        
        # Load state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
        # Setup training
        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        
        model.to(device)
        model.train()
        
        # D2) Validate max_seq_length against model's block_size
        if max_seq_length > model.config.block_size:
            yield (make_progress_html(0, 1, 'red'),
                   f"Error: max_seq_length ({max_seq_length}) exceeds model's block_size ({model.config.block_size}). "
                   f"Please reduce max_seq_length or use a model with larger context.",
                   empty_plot_data)
            return
        
        # Optimizer
        if is_self_attention_model:
            optimizer = configure_optimizers_self_attn(
                model, weight_decay=0.01, learning_rate=learning_rate, 
                betas=(0.9, 0.999), device_type=device_type
            )
        else:
            optimizer = configure_optimizers(
                model, weight_decay=0.01, learning_rate=learning_rate,
                betas=(0.9, 0.999), device_type=device_type
            )
        
        # D1) GradScaler should only be enabled for CUDA + float16
        scaler = torch.cuda.amp.GradScaler(enabled=(device_type == 'cuda' and dtype == 'float16'))
        
        # D3) Pre-filter dataset and cache tokenized results to avoid double tokenization
        # This ensures progress bar and LR schedule are accurate, and improves training efficiency
        yield (make_progress_html(0, 1), "Pre-processing dataset to filter invalid samples...", empty_plot_data)
        
        # Cache: list of (input_ids, labels) tuples for valid samples
        # Also collect detailed error statistics for debugging
        cached_tokenized_samples = []
        error_stats = {
            "no_assistant_header": 0,
            "multiple_assistant_headers": 0,
            "prompt_too_long": 0,
            "token_not_in_vocab": 0,
            "token_out_of_range": 0,
        }
        missing_token_ids = set()  # Collect unique missing token IDs
        sample_errors = []  # Collect first few sample errors for detailed reporting
        
        for i, sample in enumerate(dataset):
            result = tokenize_sft_sample(sample, tokenizer, max_seq_length, system_prompt, model_vocab_size, old2new_mapping, return_error=True)
            if result is None:
                # Unexpected: should not happen with return_error=True
                continue
            if result[0] is None:
                # Error case: result = (None, error_reason)
                error_reason = result[1]
                if error_reason.startswith("token_not_in_vocab:"):
                    error_stats["token_not_in_vocab"] += 1
                    token_id = error_reason.split(":")[1]
                    missing_token_ids.add(int(token_id))
                elif error_reason.startswith("prompt_too_long:"):
                    error_stats["prompt_too_long"] += 1
                elif error_reason.startswith("token_out_of_range:"):
                    error_stats["token_out_of_range"] += 1
                elif error_reason in error_stats:
                    error_stats[error_reason] += 1
                
                # Collect first 3 sample errors for detailed reporting
                if len(sample_errors) < 3:
                    sample_errors.append({
                        "index": i,
                        "reason": error_reason,
                        "instruction": sample.get("instruction", "")[:50] + "..." if len(sample.get("instruction", "")) > 50 else sample.get("instruction", "")
                    })
            else:
                # Success case: result = (input_ids, labels)
                cached_tokenized_samples.append(result)  # (input_ids, labels)
        
        if len(cached_tokenized_samples) == 0:
            # Build detailed error message in HTML format for better display
            error_details = []
            for err_type, count in error_stats.items():
                if count > 0:
                    error_details.append(f"<li><code>{err_type}</code>: {count}</li>")
            
            error_msg = f"<div style='font-family: monospace; padding: 10px; background: #fff3f3; border-radius: 8px; border: 1px solid #ffcccc;'>"
            error_msg += f"<b style='color: #cc0000;'>Error: No valid samples after filtering</b><br>"
            error_msg += f"<span style='color: #666;'>Total samples in dataset: {len(dataset)}</span><br><br>"
            
            error_msg += f"<b>📊 Filter Statistics:</b><br>"
            if error_details:
                error_msg += "<ul style='margin: 5px 0 10px 20px; padding: 0;'>" + "".join(error_details) + "</ul>"
            else:
                error_msg += "<span style='color: #999;'>No specific errors recorded (unexpected)</span><br><br>"
            
            if missing_token_ids:
                error_msg += f"<b>🔤 Missing Token IDs (first 10):</b><br>"
                error_msg += f"<code style='background: #f0f0f0; padding: 2px 5px;'>{sorted(list(missing_token_ids))[:10]}</code><br>"
                error_msg += f"<span style='color: #666; font-size: 0.9em;'>"
                error_msg += "This usually means the model was trained on different data and doesn't include tokens needed for SFT.<br>"
                error_msg += "</span>"
                error_msg += f"<div style='background: #e8f4fd; padding: 8px; border-radius: 4px; margin: 8px 0;'>"
                error_msg += f"<b>💡 Solution:</b> Please <b>re-process the training data</b> (go to Data Processing tab and run again). "
                error_msg += f"The latest version automatically includes SFT chat template tokens (system/user/assistant)."
                error_msg += f"</div><br>"
            
            if sample_errors:
                error_msg += f"<b>📝 Example Failed Samples:</b><br>"
                error_msg += "<ul style='margin: 5px 0 10px 20px; padding: 0;'>"
                for err in sample_errors:
                    escaped_instruction = err['instruction'].replace('<', '&lt;').replace('>', '&gt;')
                    error_msg += f"<li>Sample {err['index']}: <code>{err['reason']}</code><br>"
                    error_msg += f"<span style='color: #666; font-size: 0.9em;'>instruction: {escaped_instruction}</span></li>"
                error_msg += "</ul>"
            
            error_msg += f"<b>🔧 Diagnostic Info:</b><br>"
            error_msg += "<ul style='margin: 5px 0 10px 20px; padding: 0;'>"
            error_msg += f"<li>max_seq_length: <code>{max_seq_length}</code></li>"
            error_msg += f"<li>model_vocab_size: <code>{model_vocab_size}</code></li>"
            error_msg += f"<li>old2new_mapping size: <code>{len(old2new_mapping) if old2new_mapping else 'None'}</code></li>"
            error_msg += "</ul>"
            error_msg += "</div>"
            
            yield (make_progress_html(0, 1, 'red'), error_msg, empty_plot_data)
            return
        
        filtered_count = len(dataset) - len(cached_tokenized_samples)
        if filtered_count > 0:
            # Build filter detail message in HTML format
            filter_details = []
            for err_type, count in error_stats.items():
                if count > 0:
                    filter_details.append(f"<code>{err_type}</code>: {count}")
            detail_str = " (" + ", ".join(filter_details) + ")" if filter_details else ""
            
            missing_info = ""
            if missing_token_ids:
                missing_info = f"<br><span style='color: #666; font-size: 0.9em;'>Missing token IDs (first 5): <code>{sorted(list(missing_token_ids))[:5]}</code></span>"
            
            msg = f"<div style='font-family: monospace; padding: 8px; background: #fff9e6; border-radius: 6px; border: 1px solid #ffe066;'>"
            msg += f"⚠️ Filtered {filtered_count}/{len(dataset)} samples{detail_str}{missing_info}<br>"
            msg += f"✅ Training with <b>{len(cached_tokenized_samples)}</b> valid samples."
            msg += "</div>"
            
            yield (make_progress_html(0, 1), msg, empty_plot_data)
        
        num_samples = len(cached_tokenized_samples)
        
        # Training setup with accurate counts
        os.makedirs(out_dir, exist_ok=True)
        # num_samples already set from cached_tokenized_samples
        steps_per_epoch = math.ceil(num_samples / batch_size)
        total_micro_steps = epochs * steps_per_epoch
        # B3) Calculate optimizer steps (actual parameter updates)
        total_opt_steps = math.ceil(total_micro_steps / gradient_accumulation_steps)
        warmup_opt_steps = int(total_opt_steps * warmup_ratio)
        
        train_losses = []
        train_steps = []
        
        micro_step = 0  # Counts every forward pass
        opt_step = 0    # Counts optimizer updates (every gradient_accumulation_steps)
        
        # Create indices for shuffling (to avoid shuffling the cached data directly)
        sample_indices = list(range(num_samples))
        
        for epoch in range(epochs):
            if sft_stop_signal:
                break
            
            # Shuffle indices instead of dataset
            np.random.shuffle(sample_indices)
            
            for batch_start in range(0, num_samples, batch_size):
                if sft_stop_signal:
                    yield (make_progress_html(micro_step, total_micro_steps, 'orange'),
                           "SFT training stopped by user",
                           (train_steps, train_losses, [], []))
                    break
                
                batch_end = min(batch_start + batch_size, num_samples)
                batch_indices = sample_indices[batch_start:batch_end]
                
                # Use cached tokenized data directly instead of re-tokenizing
                batch_input_ids = [cached_tokenized_samples[i][0] for i in batch_indices]
                batch_labels = [cached_tokenized_samples[i][1] for i in batch_indices]
                
                # Pad sequences
                max_len = max(len(ids) for ids in batch_input_ids)
                padded_input_ids = [ids + [0] * (max_len - len(ids)) for ids in batch_input_ids]
                padded_labels = [lbls + [-100] * (max_len - len(lbls)) for lbls in batch_labels]
                
                input_ids = torch.tensor(padded_input_ids, dtype=torch.long)
                labels = torch.tensor(padded_labels, dtype=torch.long)
                
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                # B3) Learning rate schedule with warmup - use opt_step for proper scaling
                if opt_step < warmup_opt_steps:
                    lr = learning_rate * (opt_step + 1) / (warmup_opt_steps + 1)
                else:
                    # Cosine decay
                    progress = (opt_step - warmup_opt_steps) / max(1, total_opt_steps - warmup_opt_steps)
                    lr = learning_rate * 0.5 * (1 + math.cos(math.pi * progress))
                
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Forward pass with proper labels masking
                # A1) Don't pass labels to model - only get logits and compute loss externally
                # This avoids potential issues with model's internal loss computation
                with ctx:
                    logits, _ = model(input_ids[:, :-1])  # Only pass input_ids, not labels
                    # A2) Use reshape instead of view to avoid contiguous tensor issues
                    # Compute cross entropy loss with ignore_index=-100 for masked tokens
                    loss = F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        labels[:, 1:].reshape(-1),
                        ignore_index=-100
                    )
                
                # B1) Scale loss by gradient accumulation steps for proper gradient averaging
                scaled_loss = loss / gradient_accumulation_steps
                
                # Backward pass with scaled loss
                scaler.scale(scaled_loss).backward()
                
                micro_step += 1
                
                if micro_step % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    opt_step += 1
                
                loss_val = loss.item()  # Log unscaled loss for interpretability
                train_losses.append(loss_val)
                train_steps.append(micro_step)
                
                # Log every 10 micro steps
                if micro_step % 10 == 0:
                    progress_html = make_progress_html(micro_step, total_micro_steps)
                    log_msg = f"Epoch {epoch+1}/{epochs}, Step {micro_step}/{total_micro_steps}, OptStep {opt_step}/{total_opt_steps}, Loss: {loss_val:.4f}, LR: {lr:.2e}"
                    yield (progress_html, log_msg, (train_steps[:], train_losses[:], [], []))
        
        # B2) Handle remaining gradients that haven't been applied
        # This happens when total_micro_steps is not divisible by gradient_accumulation_steps
        # Skip if training was stopped by user to avoid applying partial gradients
        if not sft_stop_signal and micro_step % gradient_accumulation_steps != 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            opt_step += 1
        
        # If stopped by user, don't save checkpoint and exit early
        if sft_stop_signal:
            yield (make_progress_html(micro_step, total_micro_steps, 'orange'),
                   f"SFT training stopped by user at step {micro_step}/{total_micro_steps}",
                   (train_steps, train_losses, [], []))
            return
        
        # Save SFT checkpoint with token mapping for inference
        sft_ckpt_path = os.path.join(out_dir, 'ckpt_sft.pt')
        sft_checkpoint = {
            'model': model.state_dict(),
            'model_args': model_args,
            'sft_config': {
                'epochs': epochs,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'max_seq_length': max_seq_length,
                'system_prompt': system_prompt,
                'gradient_accumulation_steps': gradient_accumulation_steps,
                'total_opt_steps': opt_step,
                'total_samples': num_samples
            },
            'is_sft': True,
            'old2new_mapping': old2new_mapping,  # Token ID remapping for inference
            'new2old_mapping': {v: k for k, v in old2new_mapping.items()}  # Reverse mapping for decoding
        }
        torch.save(sft_checkpoint, sft_ckpt_path)
        
        final_msg = f"SFT training complete! Trained for {opt_step} optimizer steps on {num_samples} samples. Checkpoint saved to {sft_ckpt_path}"
        yield (make_progress_html(total_micro_steps, total_micro_steps, 'green'),
               final_msg,
               (train_steps, train_losses, [], []))
        
    except Exception as e:
        yield (make_progress_html(0, 1, 'red'),
               f"SFT training error: {str(e)}",
               empty_plot_data)


def chat_generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    system_prompt: str = "You are a helpful assistant.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    old2new_mapping: Dict[int, int] = None,
    new2old_mapping: Dict[int, int] = None
) -> Generator[str, None, None]:
    """
    Generate chat response using Qwen template.
    
    Args:
        model: The GPT model
        tokenizer: HuggingFace tokenizer
        messages: List of {"role": "user"|"assistant", "content": str}
        system_prompt: System message
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature (must be > 0)
        top_k: Top-k sampling parameter (0 means no top-k filtering)
        old2new_mapping: Mapping from original token IDs to remapped IDs (for encoding)
        new2old_mapping: Mapping from remapped IDs back to original IDs (for decoding)
    
    Yields tokens as they are generated.
    """
    # Get device from model parameters to avoid device mismatch
    # This ensures tensors are created on the same device as the model
    device = next(model.parameters()).device
    
    # Defensive checks for sampling parameters
    # Temperature must be positive to avoid division by zero
    if temperature <= 0:
        temperature = 1e-7  # Use a very small positive value for near-greedy sampling
    
    # Get vocabulary size for top_k validation
    vocab_size = model.config.vocab_size
    
    # top_k should be between 0 and vocab_size
    # 0 or negative means no top-k filtering
    # Values > vocab_size are clamped to vocab_size
    if top_k <= 0:
        top_k = 0  # Disable top-k filtering
    elif top_k > vocab_size:
        top_k = vocab_size  # Clamp to vocab size
    
    # Format conversation
    prompt = format_chat_for_inference(messages, system_prompt)
    
    # Tokenize (original token IDs from tokenizer)
    original_ids = tokenizer.encode(prompt).ids
    
    # Remap to model's token IDs if mapping is provided
    if old2new_mapping is not None:
        input_ids = []
        for i, orig_id in enumerate(original_ids):
            if orig_id in old2new_mapping:
                input_ids.append(old2new_mapping[orig_id])
            else:
                # Unknown token - raise error instead of silent fallback
                # Decoding the problematic token for better error message
                try:
                    problematic_token = tokenizer.decode([orig_id])
                except:
                    problematic_token = f"<id={orig_id}>"
                raise ValueError(
                    f"Unknown token ID {orig_id} ('{problematic_token}') at position {i} is not in model vocabulary. "
                    f"This token was not seen during training. Please ensure the input uses only characters/tokens "
                    f"that were present in the training data, or retrain with a dataset that includes this token."
                )
    else:
        input_ids = original_ids
    
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    model.eval()
    block_size = model.config.block_size
    
    # Get remapped end token IDs for stopping condition
    if old2new_mapping is not None:
        im_end_id_mapped = old2new_mapping.get(IM_END_ID, None)
        eot_id_mapped = old2new_mapping.get(EOT_ID, None)
    else:
        im_end_id_mapped = IM_END_ID
        eot_id_mapped = EOT_ID
    
    generated_text = ""
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, -1].unsqueeze(-1)] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            
            new_token_id = idx_next[0].item()
            
            # Check for end token (using remapped IDs)
            if new_token_id == im_end_id_mapped or new_token_id == eot_id_mapped:
                break
            
            # Convert back to original token ID for decoding
            if new2old_mapping is not None:
                original_token_id = new2old_mapping.get(new_token_id, new_token_id)
            else:
                original_token_id = new_token_id
            
            # Decode new token
            new_text = tokenizer.decode([original_token_id])
            generated_text += new_text
            yield new_text
    
    return generated_text
