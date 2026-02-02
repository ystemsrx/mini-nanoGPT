# src/data_process.py
import os
import pickle
import math
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import tiktoken

from src.config import DEFAULT_CONFIG, IntegerTypes
from src.db_manager import DBManager
from src.utils import compose_model_dirs

dbm = DBManager()

# =============================================================================
# SFT Chat Template Required Tokens (Qwen format)
# =============================================================================

# Special control tokens (true special tokens with dedicated IDs)
QWEN_SPECIAL_TOKEN_IDS = [
    151643,  # <|endoftext|> / EOT
    151644,  # <|im_start|>
    151645,  # <|im_end|>
]

# Chat template formatting tokens (regular tokens used in chat template structure)
# These are normal tokens that happen to be required for the chat format
QWEN_SFT_FORMAT_TOKEN_IDS = [
    8948,    # "system" - role identifier
    872,     # "user" - role identifier
    77091,   # "assistant" - role identifier
    198,     # "\n" - newline after role names (e.g., "system\n", "user\n")
]

def get_chunks(text, n):
    """
    Splits the text into 'n' roughly equal chunks for parallel processing.
    """
    chunk_size = math.ceil(len(text) / n)
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def get_unique_chars(text):
    """
    Returns a set of unique characters found in the text.
    """
    return set(text)

def encode_text_chunk(chunk, stoi):
    """
    Encodes a chunk of text at the character level using the 'stoi' dictionary.
    """
    return [stoi.get(ch, 0) for ch in chunk] # Use 0 for unknown characters

def encode_gpt2_chunk(chunk, tokenizer):
    """
    Encodes a chunk of text using GPT-2 tokenizer.
    """
    return tokenizer.encode(chunk, allowed_special={"<|endoftext|>"})

def _encode_custom_chunk(chunk: str, tokenizer_path: str):
    """
    Loads a local tokenizer from tokenizer_path (e.g., assets/tokenizer.json)
    and encodes a chunk of text.
    This function is designed to be used with multiprocessing,
    where the tokenizer is loaded in each child process.
    """
    from tokenizers import Tokenizer # Import here for multiprocessing compatibility
    tok = Tokenizer.from_file(tokenizer_path)
    return tok.encode(chunk).ids


def process_data(
    *,
    model_name: str,
    new_model: bool,
    selected_model_id: int | None = None,
    input_text: str = "",
    input_dir: str = "",
    train_split_ratio: float = DEFAULT_CONFIG["data_process"]["train_split_ratio"],
    no_validation: bool = DEFAULT_CONFIG["data_process"]["no_validation"],
    use_gpt2_tokenizer: bool = DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"],
    num_proc: int = DEFAULT_CONFIG["data_process"]["num_proc"]
):
    """
    Processes input text data for model training.

    Key steps:
    1. Determines model ID and associated file paths.
    2. Loads text from direct input or a directory.
    3. Tokenizes the text using either a custom tokenizer (if `assets/tokenizer.json` exists
       and `use_gpt2_tokenizer` is true), GPT-2 tokenizer (fallback if custom one is not found
       or if `use_gpt2_tokenizer` is true), or character-level encoding.
    4. Splits data into training and validation sets (unless `no_validation` is true).
    5. Saves processed data (token IDs) to .bin files and metadata (vocabulary, etc.) to meta.pkl.

    - If `use_gpt2_tokenizer` is checked, it first attempts to use `assets/tokenizer.json`.
      If not found, it falls back to the GPT-2 tokenizer.
    - Only tokens that actually appear in the input data are saved in the vocabulary.
      These tokens are remapped to consecutive integer IDs starting from 0.
    """
    # For new models, we defer registration until data is validated and processed
    # This prevents creating empty model entries when data processing fails
    model_id = None  # Will be set after validation for new models
    
    # Determine model_id and paths
    if new_model:
        # Don't register yet - we'll do it after validating the data
        pass
    else:
        if selected_model_id is None:
            raise ValueError("selected_model_id is required when not creating a new model.")
        model_id = selected_model_id
        info = dbm.get_model_basic_info(model_id)
        if not info:
            raise ValueError(f"Model with ID {model_id} does not exist.")
        if info["name"] != model_name:
            dbm.rename_model(model_id, model_name)
            # Update info after renaming, though not strictly necessary for current logic flow
            # info = dbm.get_model_basic_info(model_id) # Re-fetch if needed later

        # Check if new data has been provided for processing
        has_new_data = bool(input_text.strip() or (input_dir.strip() and os.path.exists(input_dir.strip())))

        if not has_new_data:
            # If only renaming the model and no new data is provided,
            # skip data processing and return existing model information.
            raw_dir, processed_dir, _ = compose_model_dirs(model_name, model_id)

            # Attempt to load existing metadata
            meta_path = os.path.join(processed_dir, 'meta.pkl')
            vocab_size = 0
            train_size = 0
            val_size = 0
            tokenizer_name = "unknown"

            if os.path.exists(meta_path):
                try:
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    vocab_size = meta.get('vocab_size', 0)
                    tokenizer_name = meta.get('tokenizer', 'unknown')
                except Exception as e:
                    print(f"Warning: Could not read meta.pkl for model {model_id}: {e}")

            # Attempt to get dataset sizes from existing .bin files
            train_bin_path = os.path.join(processed_dir, 'train.bin')
            val_bin_path = os.path.join(processed_dir, 'val.bin')

            if os.path.exists(train_bin_path):
                try:
                    train_data = np.memmap(train_bin_path, dtype=IntegerTypes, mode='r')
                    train_size = len(train_data)
                except Exception as e:
                    print(f"Warning: Could not read train.bin for model {model_id}: {e}")

            if os.path.exists(val_bin_path):
                try:
                    val_data = np.memmap(val_bin_path, dtype=IntegerTypes, mode='r')
                    val_size = len(val_data)
                except Exception as e:
                    print(f"Warning: Could not read val.bin for model {model_id}: {e}")

            res = {
                "model_id": model_id,
                "processed_data_dir": processed_dir,
                "vocab_size": vocab_size,
                "train_size": train_size,
                "tokenizer": tokenizer_name
            }
            if val_size > 0:
                res["val_size"] = val_size
            return res

    # Load text data BEFORE registering new model to validate data first
    data = input_text.strip()
    if not data and input_dir.strip():
        input_dir_abs = input_dir.strip()
        if os.path.exists(input_dir_abs):
            # Concatenate content from all .txt files in the specified directory
            for fn in (f for f in os.listdir(input_dir_abs) if f.endswith(".txt")):
                try:
                    with open(os.path.join(input_dir_abs, fn), "r", encoding="utf-8") as f_in:
                        data += f_in.read()
                except Exception as e:
                    print(f"Warning: Could not read file {fn}: {e}")
    
    # Validate data BEFORE registering new model to database
    if not data:
        raise ValueError("No input text provided. Please check your input text or directory.")

    # Ensure data ends with a newline character for SFT compatibility
    if not data.endswith('\n'):
        data += '\n'

    # Now that data is validated, register new model if needed
    if new_model:
        model_id = dbm.register_model(model_name)

    raw_dir, processed_dir, _ = compose_model_dirs(model_name, model_id)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # Save the combined raw input text
    with open(os.path.join(raw_dir, "merged_input.txt"), "w", encoding="utf-8") as f_out:
        f_out.write(data)

    # Tokenize and split data
    size_mb = len(data.encode("utf-8")) / 1024 / 1024
    # Use single process for small files to avoid overhead of multiprocessing
    actual_proc = min(num_proc, cpu_count()) if size_mb > 100 else 1

    # Use tokenizer (assets/tokenizer.json or GPT-2)
    if use_gpt2_tokenizer:
        tokenizer_path = Path.cwd() / "assets/tokenizer.json"

        # If assets/tokenizer.json exists in the root directory, use HuggingFace Tokenizers
        if tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer # Check for 'tokenizers' library dependency
            except ImportError as e:
                raise ImportError(
                    "Detected assets/tokenizer.json, but the `tokenizers` library is not installed. "
                    "Please install it by running: pip install tokenizers"
                ) from e

            tok_name = "custom_json" # Tokenizer type identifier
            chunks = get_chunks(data, actual_proc) if actual_proc > 1 else [data]

            if actual_proc == 1:
                # Single process encoding
                tokenizer_instance = Tokenizer.from_file(str(tokenizer_path))
                token_chunks = [tokenizer_instance.encode(c).ids for c in chunks]
            else:
                # Multi-process encoding: Load Tokenizer in child processes
                with Pool(actual_proc) as pool:
                    token_chunks = pool.starmap(
                        _encode_custom_chunk,
                        [(c, str(tokenizer_path)) for c in chunks]
                    )
            tokens_full = [t for ck in token_chunks for t in ck] # Flatten list of token lists

            # Append an end-of-text token if it's defined in the tokenizer and not already present
            eot_id_old = None
            # Check for common EOT token representations
            for special_token_str in ["<|endoftext|>", "</s>", "[EOS]"]: # Add other common EOTs if needed
                try:
                    # Temporary tokenizer instance to get token ID
                    temp_tokenizer = Tokenizer.from_file(str(tokenizer_path))
                    eot_id_old = temp_tokenizer.token_to_id(special_token_str)
                    if eot_id_old is not None:
                        break
                except Exception: # Handles cases where token_to_id might fail or token doesn't exist
                    pass
            
            if eot_id_old is not None and (not tokens_full or tokens_full[-1] != eot_id_old):
                tokens_full.append(eot_id_old)

        # If assets/tokenizer.json is not found, fallback to GPT-2 (tiktoken)
        else:
            enc = tiktoken.get_encoding("gpt2")
            tok_name = "gpt2" # Tokenizer type identifier
            chunks = get_chunks(data, actual_proc) if actual_proc > 1 else [data]
            if actual_proc == 1:
                token_chunks = [encode_gpt2_chunk(c, enc) for c in chunks]
            else:
                with Pool(actual_proc) as pool:
                    token_chunks = pool.starmap(encode_gpt2_chunk, [(c, enc) for c in chunks])
            tokens_full = [t for ck in token_chunks for t in ck] # Flatten
            
            # Ensure GPT-2 EOT token is appended if not present
            if not tokens_full or tokens_full[-1] != enc.eot_token:
                tokens_full.append(enc.eot_token)
            eot_id_old = enc.eot_token # Original EOT token ID from GPT-2

        # Simplify the subword vocabulary: map original token IDs to new consecutive IDs (0, 1, 2, ...)
        # This ensures the vocabulary only contains tokens actually present in the dataset,
        # PLUS tokens required for SFT chat template formatting.
        used_old_ids_set = set(tokens_full)
        
        # For custom tokenizer (Qwen), include all tokens required for SFT compatibility
        if tok_name == "custom_json":
            # Add special control tokens (<|im_start|>, <|im_end|>, <|endoftext|>)
            for special_id in QWEN_SPECIAL_TOKEN_IDS:
                used_old_ids_set.add(special_id)
            # Add chat template formatting tokens (role names + newline)
            for format_id in QWEN_SFT_FORMAT_TOKEN_IDS:
                used_old_ids_set.add(format_id)
        
        used_old_ids = sorted(list(used_old_ids_set)) # Unique sorted original token IDs
        old2new = {old_id: new_id for new_id, old_id in enumerate(used_old_ids)}
        tokens = [old2new[t] for t in tokens_full] # Remapped token sequence

        if no_validation:
            splits = {"train": tokens}
            # Remove any existing old validation set file if we are not creating one now
            val_bin_path = os.path.join(processed_dir, "val.bin")
            if os.path.exists(val_bin_path):
                try:
                    os.remove(val_bin_path)
                    print(f"Removed old validation set file: {val_bin_path}")
                except OSError as e:
                    print(f"Warning: Could not remove old validation set file {val_bin_path}: {e}")
        else:
            split_at = int(len(tokens) * train_split_ratio)
            splits = {"train": tokens[:split_at], "val": tokens[split_at:]}

        # Save token sequences to .bin files
        for sp, seq in splits.items():
            # Using uint32 for tokens as vocab sizes are typically < 2^32
            # Consider IntegerTypes if a different type is consistently needed
            np.array(seq, dtype=np.uint32).tofile(os.path.join(processed_dir, f"{sp}.bin"))

        # Build meta.pkl content
        if tokenizer_path.exists() and tok_name == "custom_json":
            # For a custom assets/tokenizer.json, use HuggingFace Tokenizer's decode method
            tokenizer_for_meta = Tokenizer.from_file(str(tokenizer_path))
            # Create itos (ID to string) mapping using the new consecutive IDs
            itos = {new_id: tokenizer_for_meta.decode([old_id]) for old_id, new_id in old2new.items()}
        else:
            # For GPT-2 fallback, use tiktoken's decode method
            # Ensure enc is defined (it will be if tok_name is "gpt2")
            itos = {new_id: enc.decode([old_id]) for old_id, new_id in old2new.items()}

        stoi = {s: i for i, s in itos.items()} # String to ID mapping
        vocab_size = len(itos)

        meta = {
            "vocab_size": vocab_size,
            "itos": itos,
            "stoi": stoi,
            "tokenizer": tok_name,
            "old2new_mapping": old2new, # Store the mapping from original to new token IDs
            "eot_token_new_id": old2new.get(eot_id_old, None) # Store the new ID for the EOT token
        }
        train_sz = len(splits["train"])
        val_sz = len(splits.get("val", []))

    # Character-level encoding
    else:
        tok_name = "character_level" # Tokenizer type identifier
        # Determine unique characters across all chunks (if multiprocessing)
        if actual_proc == 1:
            chars = sorted(list(get_unique_chars(data)))
        else:
            with Pool(actual_proc) as pool:
                char_sets = pool.map(get_unique_chars, get_chunks(data, actual_proc))
            chars = sorted(list(set().union(*char_sets)))

        stoi = {ch: i for i, ch in enumerate(chars)} # Character to ID
        itos = {i: ch for ch, i in stoi.items()}    # ID to character
        vocab_size = len(chars)

        # Encode the entire dataset
        if actual_proc == 1:
            encoded = encode_text_chunk(data, stoi)
        else:
            with Pool(actual_proc) as pool:
                enc_chunks = pool.starmap(encode_text_chunk, [(c, stoi) for c in get_chunks(data, actual_proc)])
            encoded = [e for ck in enc_chunks for e in ck] # Flatten

        if no_validation:
            train_ids = np.array(encoded, dtype=IntegerTypes)
            val_ids = None # Explicitly set to None
            # Remove any existing old validation set file
            val_bin_path = os.path.join(processed_dir, "val.bin")
            if os.path.exists(val_bin_path):
                try:
                    os.remove(val_bin_path)
                    print(f"Removed old validation set file: {val_bin_path}")
                except OSError as e:
                    print(f"Warning: Could not remove old validation set file {val_bin_path}: {e}")
        else:
            split_at = int(len(encoded) * train_split_ratio)
            train_ids = np.array(encoded[:split_at], dtype=IntegerTypes)
            val_ids = np.array(encoded[split_at:], dtype=IntegerTypes)

        # Save processed data
        train_ids.tofile(os.path.join(processed_dir, "train.bin"))
        if val_ids is not None:
            val_ids.tofile(os.path.join(processed_dir, "val.bin"))

        # Add tokenizer information to meta
        meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi, "tokenizer": tok_name}
        train_sz = len(train_ids)
        val_sz = len(val_ids) if val_ids is not None else 0

    # Save meta.pkl
    with open(os.path.join(processed_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # Return results
    res = {
        "model_id": model_id,
        "processed_data_dir": processed_dir,
        "vocab_size": vocab_size,
        "train_size": train_sz,
        "tokenizer": tok_name # Name of the tokenizer used
    }
    if not no_validation and val_sz > 0 : # Ensure val_sz is positive if not no_validation
        res["val_size"] = val_sz

    return res
