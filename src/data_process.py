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
    return [stoi.get(ch, 0) for ch in chunk]

def encode_gpt2_chunk(chunk, tokenizer):
    """
    Encodes a chunk of text using GPT-2 tokenizer.
    """
    return tokenizer.encode(chunk, allowed_special={"<|endoftext|>"})

# ------ HuggingFace Tokenizers ------
def _encode_custom_chunk(chunk: str, tokenizer_path: str):
    """
    Load local assets/tokenizer.json and encode a chunk of text.
    """
    from tokenizers import Tokenizer        # 局部 import，主进程无需强依赖
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
    - If "Use tokenizer" is checked, it will first attempt to use `assets/tokenizer.json` in the root directory;
      if it does not exist, it will fall back to the GPT-2 tokenizer.
    - Only **actually occurring** tokens are saved, and they are automatically remapped to consecutive IDs.
    """
    # -------- 0. 决定 model_id & 路径 -------- #
    if new_model:
        model_id = dbm.register_model(model_name)
    else:
        if selected_model_id is None:
            raise ValueError("selected_model_id is empty, you can create a new model.")
        model_id = selected_model_id
        info = dbm.get_model_basic_info(model_id)
        if not info:
            raise ValueError(f"Model {model_id} does not exist.")
        if info["name"] != model_name:
            dbm.rename_model(model_id, model_name)
            # Update info after renaming
            info = dbm.get_model_basic_info(model_id)
        
        # Check if we have new data to process
        has_new_data = bool(input_text.strip() or (input_dir.strip() and os.path.exists(input_dir.strip())))
        
        if not has_new_data:
            # Only renaming, no data processing needed
            # Return existing model information
            raw_dir, processed_dir, _ = compose_model_dirs(model_name, model_id)
            
            # Try to get existing vocab_size from meta.pkl
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
                    print(f"Warning: Could not read meta.pkl: {e}")
            
            # Try to get dataset sizes from existing bin files
            train_bin_path = os.path.join(processed_dir, 'train.bin')
            val_bin_path = os.path.join(processed_dir, 'val.bin')
            
            if os.path.exists(train_bin_path):
                try:
                    train_data = np.memmap(train_bin_path, dtype=IntegerTypes, mode='r')
                    train_size = len(train_data)
                except Exception as e:
                    print(f"Warning: Could not read train.bin: {e}")
            
            if os.path.exists(val_bin_path):
                try:
                    val_data = np.memmap(val_bin_path, dtype=IntegerTypes, mode='r')
                    val_size = len(val_data)
                except Exception as e:
                    print(f"Warning: Could not read val.bin: {e}")
            
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

    raw_dir, processed_dir, _ = compose_model_dirs(model_name, model_id)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # -------- 1. 读取文本 -------- #
    data = input_text.strip()
    if not data and input_dir.strip():
        input_dir_abs = input_dir.strip()
        if os.path.exists(input_dir_abs):
            for fn in (f for f in os.listdir(input_dir_abs) if f.endswith(".txt")):
                with open(os.path.join(input_dir_abs, fn), "r", encoding="utf-8") as f:
                    data += f.read()
    if not data:
        raise ValueError("It seems that you haven't provided any text. Please check your input.")

    with open(os.path.join(raw_dir, "merged_input.txt"), "w", encoding="utf-8") as f:
        f.write(data)

    # -------- 2. Tokenize & 切分 -------- #
    size_mb = len(data.encode("utf-8")) / 1024 / 1024
    actual_proc = min(num_proc, cpu_count()) if size_mb > 100 else 1

    # ================================
    # 2-A. 使用分词器 (assets/tokenizer.json / GPT-2)
    # ================================
    if use_gpt2_tokenizer:
        tokenizer_path = Path.cwd() / "assets/tokenizer.json"

        # ---- ① 根目录存在 assets/tokenizer.json → 使用 HuggingFace Tokenizers ----
        if tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer  # 提前检测依赖
            except ImportError as e:
                raise ImportError(
                    "Detected assets/tokenizer.json, but the `tokenizers` library is not installed in the current environment:\n"
                    "    pip install tokenizers\n"
                ) from e

            tok_name = "custom_json"
            chunks = get_chunks(data, actual_proc) if actual_proc > 1 else [data]
            if actual_proc == 1:
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                token_chunks = [tokenizer.encode(c).ids for c in chunks]
            else:
                # Multi-process: Load Tokenizer in child processes
                with Pool(actual_proc) as pool:
                    token_chunks = pool.starmap(
                        _encode_custom_chunk,
                        [(c, str(tokenizer_path)) for c in chunks]
                    )
            tokens_full = [t for ck in token_chunks for t in ck]

            # Append <|endoftext|> or other common end tokens if they exist
            eot_id_old = None
            for special in ["", "<|endoftext|>"]:
                try:
                    test_tok = Tokenizer.from_file(str(tokenizer_path))
                    eot_id_old = test_tok.token_to_id(special)
                    if eot_id_old is not None:
                        break
                except Exception:
                    pass
            if eot_id_old is not None and (len(tokens_full) == 0 or tokens_full[-1] != eot_id_old):
                tokens_full.append(eot_id_old)

        # ---- ② No assets/tokenizer.json → Fallback to GPT-2 (tiktoken) ----
        else:
            enc = tiktoken.get_encoding("gpt2")
            tok_name = "gpt2"
            chunks = get_chunks(data, actual_proc) if actual_proc > 1 else [data]
            with Pool(actual_proc) as pool:
                token_chunks = pool.starmap(encode_gpt2_chunk, [(c, enc) for c in chunks])
            tokens_full = [t for ck in token_chunks for t in ck]
            if tokens_full[-1] != enc.eot_token:
                tokens_full.append(enc.eot_token)
            eot_id_old = enc.eot_token

        # ---- ③ Simplify the subword vocabulary: old-id → new-id ----
        used_old_ids = sorted(set(tokens_full))
        old2new = {old: new for new, old in enumerate(used_old_ids)}
        tokens = [old2new[t] for t in tokens_full]

        if no_validation:
            splits = {"train": tokens}
        else:
            split_at = int(len(tokens) * train_split_ratio)
            splits = {"train": tokens[:split_at], "val": tokens[split_at:]}

        for sp, seq in splits.items():
            np.array(seq, dtype=np.uint32).tofile(os.path.join(processed_dir, f"{sp}.bin"))

        # ---- ④ Build meta.pkl ----
        if tokenizer_path.exists():
            # For custom assets/tokenizer.json，use HF Tokenizers decode
            tokenizer_for_meta = Tokenizer.from_file(str(tokenizer_path))
            itos = {nid: tokenizer_for_meta.decode([oid]) for oid, nid in old2new.items()}
        else:
            # GPT-2 fallback
            itos = {nid: enc.decode([oid]) for oid, nid in old2new.items()}

        stoi = {s: i for i, s in itos.items()}
        vocab_size = len(itos)

        meta = {
            "vocab_size": vocab_size,
            "itos": itos,
            "stoi": stoi,
            "tokenizer": tok_name,
            "old2new": old2new,
            "eot_id_new": old2new.get(eot_id_old, None)
        }
        train_sz = len(splits["train"])
        val_sz = len(splits.get("val", []))

    # ================================
    # 2-B. Character-level encoding
    # ================================
    else:
        tok_name = "character level"
        chars = sorted(set(data)) if actual_proc == 1 else sorted(
            set().union(*Pool(actual_proc).map(get_unique_chars, get_chunks(data, actual_proc)))
        )
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        vocab_size = len(chars)

        if actual_proc == 1:
            encoded = encode_text_chunk(data, stoi)
        else:
            with Pool(actual_proc) as pool:
                enc_chunks = pool.starmap(encode_text_chunk, [(c, stoi) for c in get_chunks(data, actual_proc)])
            encoded = [e for ck in enc_chunks for e in ck]

        if no_validation:
            train_ids = np.array(encoded, dtype=IntegerTypes)
            val_ids = None
        else:
            split_at = int(len(encoded) * train_split_ratio)
            train_ids = np.array(encoded[:split_at], dtype=IntegerTypes)
            val_ids = np.array(encoded[split_at:], dtype=IntegerTypes)

        train_ids.tofile(os.path.join(processed_dir, "train.bin"))
        if val_ids is not None:
            val_ids.tofile(os.path.join(processed_dir, "val.bin"))

        meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi, "tokenizer": tok_name}  # 添加tokenizer信息到meta
        train_sz = len(train_ids)
        val_sz = len(val_ids) if val_ids is not None else 0

    # -------- 3. Save meta.pkl -------- #
    with open(os.path.join(processed_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    # -------- 4. Return -------- #
    res = {
        "model_id": model_id,
        "processed_data_dir": processed_dir,
        "vocab_size": vocab_size,
        "train_size": train_sz,
        "tokenizer": tok_name
    }
    if not no_validation:
        res["val_size"] = val_sz
        
    return res
