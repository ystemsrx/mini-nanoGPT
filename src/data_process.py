# src/data_process.py
import os
import pickle
import math
import json
import random
from multiprocessing import Pool, cpu_count
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial

import numpy as np

from src.config import DEFAULT_CONFIG, IntegerTypes

# Global tokenizer instance for multiprocessing (initialized by pool initializer)
_global_tokenizer = None
_global_tokenizer_path = None
from src.db_manager import DBManager
from src.utils import compose_model_dirs

dbm = DBManager()

# =============================================================================
# Supported Data Formats
# =============================================================================
SUPPORTED_EXTENSIONS = {'.txt', '.jsonl'}


def load_text_file(file_path: Path) -> list[str]:
    """
    Load a plain text file and return its content as a single document.
    
    Args:
        file_path: Path to the .txt file
        
    Returns:
        List containing the file content as a single document
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if content:
                return [content]
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
    return []


def _init_tokenizer_pool(tokenizer_path: str):
    """
    Initializer function for multiprocessing Pool.
    Loads the tokenizer once per worker process.
    """
    global _global_tokenizer, _global_tokenizer_path
    from tokenizers import Tokenizer
    _global_tokenizer = Tokenizer.from_file(tokenizer_path)
    _global_tokenizer_path = tokenizer_path


def _parse_json_line(line_data: tuple) -> str | None:
    """
    Parse a single JSON line (for parallel processing).
    
    Args:
        line_data: Tuple of (line_num, line_content, file_path_str)
        
    Returns:
        Extracted text or None if parsing failed
    """
    line_num, line, file_path_str = line_data
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
        if isinstance(obj, dict) and 'text' in obj:
            text = obj['text']
            if isinstance(text, str) and text.strip():
                return text.strip()
    except json.JSONDecodeError:
        pass
    return None


# Threshold for enabling parallel JSON parsing within a single JSONL file
# Files with more lines than this will use ThreadPool for parsing
JSONL_PARALLEL_LINE_THRESHOLD = 5000


def _load_and_parse_jsonl_file(file_path: str) -> tuple[list[str], str]:
    """
    Load and parse a JSONL file with parallel JSON parsing for large files.
    This function is designed for multiprocessing - it handles the entire file in one process.
    
    Args:
        file_path: Path to the .jsonl file
        
    Returns:
        Tuple of (documents list, file_type string)
    """
    file_path = Path(file_path)
    documents = []
    
    try:
        # Read all lines first
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # For large files, use thread pool for parallel JSON parsing
        # ThreadPool is efficient here because JSON parsing releases GIL during string operations
        if len(lines) > JSONL_PARALLEL_LINE_THRESHOLD:
            line_data = [(i + 1, line, str(file_path)) for i, line in enumerate(lines)]
            
            # Use ThreadPoolExecutor for CPU-bound JSON parsing within this process
            # Limit workers to avoid excessive context switching
            num_workers = min(4, cpu_count())
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(executor.map(_parse_json_line, line_data))
            
            documents = [r for r in results if r is not None]
        else:
            # For smaller files, parse sequentially (avoid thread overhead)
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict) and 'text' in obj:
                        text = obj['text']
                        if isinstance(text, str) and text.strip():
                            documents.append(text.strip())
                except json.JSONDecodeError:
                    pass
                    
    except Exception as e:
        print(f"Warning: Could not read file {file_path}: {e}")
    
    return (documents, 'jsonl')


def _load_text_file_for_pool(file_path: str) -> tuple[list[str], str]:
    """
    Load a text file (wrapper for multiprocessing).
    
    Args:
        file_path: Path to the .txt file
        
    Returns:
        Tuple of (documents list, file_type string)
    """
    docs = load_text_file(Path(file_path))
    return (docs, 'txt')


def load_jsonl_file(file_path: Path) -> list[str]:
    """
    Load a JSONL file where each line contains a JSON object with a "text" field.
    Each line is treated as a separate document that will get an EOT token appended.
    
    Expected format:
        {"text": "document content 1"}
        {"text": "document content 2"}
        ...
    
    Args:
        file_path: Path to the .jsonl file
        
    Returns:
        List of text contents extracted from the JSONL file
    """
    documents, _ = _load_and_parse_jsonl_file(str(file_path))
    return documents


def load_documents_from_directory(directory: str | Path, num_proc: int = 1) -> tuple[list[tuple[list[str], str]], dict]:
    """
    Recursively load all supported files from a directory and its subdirectories.
    Uses parallel processing for multiple files when num_proc > 1.
    
    Supported formats:
        - .txt: Plain text files (entire file as one document)
        - .jsonl: JSON Lines files with {"text": "..."} format (each line as one document)
    
    Args:
        directory: Path to the directory to scan
        num_proc: Number of processes to use for parallel file loading
        
    Returns:
        A tuple of (file_documents, stats) where:
            - file_documents: List of (documents, file_type) tuples, each representing one file
              where documents is a list of text strings and file_type is 'txt' or 'jsonl'
            - stats: Dictionary with file processing statistics
    """
    directory = Path(directory)
    file_documents = []  # List of (documents, file_type) tuples
    stats = {
        'total_files': 0,
        'txt_files': 0,
        'jsonl_files': 0,
        'failed_files': 0,
        'total_documents': 0
    }
    
    # Collect all file paths first
    txt_files = []
    jsonl_files = []
    
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(directory.rglob(f'*{ext}')):
            if not file_path.is_file():
                continue
            if ext == '.txt':
                txt_files.append(str(file_path))
            elif ext == '.jsonl':
                jsonl_files.append(str(file_path))
    
    all_files = txt_files + jsonl_files
    stats['total_files'] = len(all_files)
    
    if not all_files:
        return file_documents, stats
    
    # Determine actual number of processes to use
    actual_proc = min(num_proc, len(all_files), cpu_count())
    
    # Process files in parallel if we have multiple files and multiple processes
    if actual_proc > 1 and len(all_files) > 1:
        print(f"Loading {len(all_files)} files using {actual_proc} processes...")
        
        # Prepare tasks: (file_path, loader_function)
        tasks = []
        for f in txt_files:
            tasks.append((f, 'txt'))
        for f in jsonl_files:
            tasks.append((f, 'jsonl'))
        
        # Use ProcessPoolExecutor for true parallel file loading
        with ProcessPoolExecutor(max_workers=actual_proc) as executor:
            # Submit tasks based on file type
            futures = []
            for file_path, file_type in tasks:
                if file_type == 'txt':
                    futures.append(executor.submit(_load_text_file_for_pool, file_path))
                else:
                    futures.append(executor.submit(_load_and_parse_jsonl_file, file_path))
            
            # Collect results
            for future in futures:
                try:
                    file_docs, file_type = future.result()
                    if file_docs:
                        file_documents.append((file_docs, file_type))
                        if file_type == 'txt':
                            stats['txt_files'] += 1
                        else:
                            stats['jsonl_files'] += 1
                    else:
                        stats['failed_files'] += 1
                except Exception as e:
                    print(f"Warning: Failed to process file: {e}")
                    stats['failed_files'] += 1
    else:
        # Sequential processing for single file or single process
        for file_path in txt_files:
            file_docs = load_text_file(Path(file_path))
            if file_docs:
                stats['txt_files'] += 1
                file_documents.append((file_docs, 'txt'))
            else:
                stats['failed_files'] += 1
        
        for file_path in jsonl_files:
            file_docs = load_jsonl_file(Path(file_path))
            if file_docs:
                stats['jsonl_files'] += 1
                file_documents.append((file_docs, 'jsonl'))
            else:
                stats['failed_files'] += 1
    
    stats['total_documents'] = sum(len(docs) for docs, _ in file_documents)
    
    # Print summary
    print(f"Directory scan complete:")
    print(f"  - Total files found: {stats['total_files']}")
    print(f"  - TXT files processed: {stats['txt_files']}")
    print(f"  - JSONL files processed: {stats['jsonl_files']}")
    print(f"  - Failed files: {stats['failed_files']}")
    print(f"  - Total documents extracted: {stats['total_documents']}")
    
    return file_documents, stats

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

def _encode_custom_chunk(chunk: str):
    """
    Encodes a chunk of text using the global tokenizer.
    This function is designed to be used with multiprocessing,
    where the tokenizer is pre-loaded by the pool initializer.
    """
    global _global_tokenizer
    return _global_tokenizer.encode(chunk).ids


def _encode_document_custom(args: tuple):
    """
    Encode a single document using custom tokenizer (for multiprocessing).
    Uses the global tokenizer pre-loaded by pool initializer.
    Returns (doc_tokens, eot_id) where eot_id should be appended after doc_tokens.
    """
    global _global_tokenizer
    doc, eot_id = args
    doc_tokens = _global_tokenizer.encode(doc).ids
    return (doc_tokens, eot_id)


def _encode_document_char(args: tuple):
    """
    Encode a single document at character level (for multiprocessing).
    Returns (doc_tokens, eot_id) where eot_id should be appended after doc_tokens.
    """
    doc, stoi, eot_id = args
    doc_tokens = [stoi.get(ch, 0) for ch in doc]
    return (doc_tokens, eot_id)


def _get_unique_chars_from_doc(doc: str):
    """
    Get unique characters from a single document (for multiprocessing).
    """
    return set(doc)


def process_data(
    *,
    model_name: str,
    new_model: bool,
    selected_model_id: int | None = None,
    input_text: str = "",
    input_dir: str = "",
    train_split_ratio: float = DEFAULT_CONFIG["data_process"]["train_split_ratio"],
    no_validation: bool = DEFAULT_CONFIG["data_process"]["no_validation"],
    use_custom_tokenizer: bool = DEFAULT_CONFIG["data_process"]["use_custom_tokenizer"],
    num_proc: int = DEFAULT_CONFIG["data_process"]["num_proc"]
):
    """
    Processes input text data for model training.

    Key steps:
    1. Determines model ID and associated file paths.
    2. Loads text from direct input or a directory.
    3. Tokenizes the text using either a custom tokenizer (if `assets/tokenizer.json` exists
       and `use_custom_tokenizer` is true), or character-level encoding.
    4. Splits data into training and validation sets (unless `no_validation` is true).
    5. Saves processed data (token IDs) to .bin files and metadata (vocabulary, etc.) to meta.pkl.

    - If `use_custom_tokenizer` is checked, it uses `assets/tokenizer.json`.
    - Only tokens that actually appear in the input data are saved in the vocabulary.
      These tokens are remapped to consecutive integer IDs starting from 0.
    """
    
    random.seed(42)
    
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
    # Use a list to track individual documents for proper EOT token insertion
    # file_documents: List of (documents, file_type) tuples for per-file processing
    file_documents = []  # List of (documents_list, file_type) tuples
    documents = []  # Flat list of all documents (for backward compatibility)
    is_multi_document = False  # Flag to indicate if we have multiple documents from directory
    
    if input_text.strip():
        documents.append(input_text.strip())
        file_documents.append(([input_text.strip()], 'txt'))  # Treat direct input as txt
    elif input_dir.strip():
        input_dir_abs = input_dir.strip()
        if os.path.exists(input_dir_abs):
            # Recursively load all supported files (.txt, .jsonl) from directory and subdirectories
            # Use parallel file loading with num_proc processes
            file_documents, load_stats = load_documents_from_directory(input_dir_abs, num_proc=num_proc)
            # Flatten to documents list for backward compatibility
            for docs, _ in file_documents:
                documents.extend(docs)
            is_multi_document = len(documents) > 1
    
    # Validate data BEFORE registering new model to database
    if not documents:
        raise ValueError("No input text provided. Please check your input text or directory.")
    
    # Concatenate for backward compatibility (raw data saving, size calculation, etc.)
    data = '\n'.join(documents)

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
    num_documents = len(documents)
    
    # Determine actual number of processes for tokenization
    # Enable parallel processing when:
    # 1. Data size > 10MB (enough data to justify process overhead), OR
    # 2. Multi-document mode with > 100 documents (many small docs benefit from parallelism)
    # Limit by num_proc setting and available CPUs
    if size_mb > 10 or (is_multi_document and num_documents > 100):
        actual_proc = min(num_proc, cpu_count())
    else:
        actual_proc = 1

    # Use custom tokenizer (assets/tokenizer.json)
    if use_custom_tokenizer:
        tokenizer_path = Path.cwd() / "assets/tokenizer.json"

        # Check if assets/tokenizer.json exists
        if not tokenizer_path.exists():
            raise FileNotFoundError(
                "Custom tokenizer enabled but assets/tokenizer.json not found. "
                "Please place your tokenizer.json file in the assets directory, "
                "or uncheck 'Use custom tokenizer' to use character-level encoding."
            )
        
        try:
            from tokenizers import Tokenizer  # Check for 'tokenizers' library dependency
        except ImportError as e:
            raise ImportError(
                "Detected assets/tokenizer.json, but the `tokenizers` library is not installed. "
                "Please install it by running: pip install tokenizers"
            ) from e

        tok_name = "custom_json"  # Tokenizer type identifier
        
        # First, determine the EOT token ID
        eot_id_old = None
        for special_token_str in ["<|endoftext|>", "</s>", "[EOS]"]:  # Check common EOT representations
            try:
                temp_tokenizer = Tokenizer.from_file(str(tokenizer_path))
                eot_id_old = temp_tokenizer.token_to_id(special_token_str)
                if eot_id_old is not None:
                    break
            except Exception:
                pass
        
        # Tokenize documents with EOT token at the end of each document
        tokenizer_instance = Tokenizer.from_file(str(tokenizer_path))
        
        # For multi-document mode with validation, we use hybrid stratified sampling:
        # - JSONL files: randomly shuffle documents, then split by document count (preserves document integrity)
        # - TXT files: split by token ratio (each document is split proportionally)
        # This ensures each data source is represented in both train and val sets.
        use_stratified_split = is_multi_document and not no_validation
        
        if use_stratified_split:
            # Stratified sampling mode: process each file independently based on its type
            train_tokens_full = []
            val_tokens_full = []
            
            # Process each file separately
            for file_docs, file_type in file_documents:
                if not file_docs:
                    continue
                
                # Tokenize all documents in this file
                min_docs_for_parallel = max(actual_proc * 2, 10)
                if actual_proc > 1 and len(file_docs) >= min_docs_for_parallel:
                    # Parallel tokenization
                    args_list = [(doc, eot_id_old) for doc in file_docs]
                    with Pool(actual_proc, initializer=_init_tokenizer_pool, initargs=(str(tokenizer_path),)) as pool:
                        results = pool.map(_encode_document_custom, args_list)
                    file_token_lists = [(doc_tokens, eot_id) for doc_tokens, eot_id in results]
                else:
                    # Sequential tokenization
                    file_token_lists = []
                    for doc in file_docs:
                        doc_tokens = tokenizer_instance.encode(doc).ids
                        file_token_lists.append((doc_tokens, eot_id_old))
                
                if file_type == 'jsonl':
                    # JSONL files: randomly assign whole documents to train/val sets
                    # This preserves document integrity (no mid-document splits)
                    indices = list(range(len(file_token_lists)))
                    random.shuffle(indices)
                    split_idx = int(len(indices) * train_split_ratio)
                    train_indices = set(indices[:split_idx])
                    
                    for i, (doc_tokens, eot_id) in enumerate(file_token_lists):
                        if not doc_tokens:
                            continue
                        doc_tokens_with_eot = list(doc_tokens)
                        if eot_id is not None:
                            doc_tokens_with_eot.append(eot_id)
                        
                        if i in train_indices:
                            train_tokens_full.extend(doc_tokens_with_eot)
                        else:
                            val_tokens_full.extend(doc_tokens_with_eot)
                else:
                    # TXT files: split each document by token ratio
                    for doc_tokens, eot_id in file_token_lists:
                        if not doc_tokens:
                            continue
                        doc_tokens_with_eot = list(doc_tokens)
                        if eot_id is not None:
                            doc_tokens_with_eot.append(eot_id)
                        
                        # Split this document by ratio
                        split_at = int(len(doc_tokens_with_eot) * train_split_ratio)
                        train_tokens_full.extend(doc_tokens_with_eot[:split_at])
                        val_tokens_full.extend(doc_tokens_with_eot[split_at:])
            
            tokens_full = train_tokens_full + val_tokens_full  # For vocabulary building
            
        elif is_multi_document:
            # Multi-document mode without validation: just concatenate all
            tokens_full = []
            min_docs_for_parallel = max(actual_proc * 2, 10)
            if actual_proc > 1 and len(documents) >= min_docs_for_parallel:
                args_list = [(doc, eot_id_old) for doc in documents]
                with Pool(actual_proc, initializer=_init_tokenizer_pool, initargs=(str(tokenizer_path),)) as pool:
                    results = pool.map(_encode_document_custom, args_list)
                for doc_tokens, eot_id in results:
                    tokens_full.extend(doc_tokens)
                    if eot_id is not None:
                        tokens_full.append(eot_id)
            else:
                for doc in documents:
                    doc_tokens = tokenizer_instance.encode(doc).ids
                    tokens_full.extend(doc_tokens)
                    if eot_id_old is not None:
                        tokens_full.append(eot_id_old)
        else:
            # Single document mode: use chunking for large files, add EOT only at the end
            chunks = get_chunks(data, actual_proc) if actual_proc > 1 else [data]
            if actual_proc == 1:
                token_chunks = [tokenizer_instance.encode(c).ids for c in chunks]
            else:
                with Pool(actual_proc, initializer=_init_tokenizer_pool, initargs=(str(tokenizer_path),)) as pool:
                    token_chunks = pool.map(_encode_custom_chunk, chunks)
            tokens_full = [t for ck in token_chunks for t in ck]
            if eot_id_old is not None and (not tokens_full or tokens_full[-1] != eot_id_old):
                tokens_full.append(eot_id_old)

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
        elif use_stratified_split:
            # Stratified sampling: use pre-split train/val tokens (already split per-document)
            train_tokens_remapped = [old2new[t] for t in train_tokens_full]
            val_tokens_remapped = [old2new[t] for t in val_tokens_full]
            splits = {"train": train_tokens_remapped, "val": val_tokens_remapped}
        else:
            # Global split for single document mode
            split_at = int(len(tokens) * train_split_ratio)
            splits = {"train": tokens[:split_at], "val": tokens[split_at:]}

        # Save token sequences to .bin files
        for sp, seq in splits.items():
            # Using uint32 for tokens as vocab sizes are typically < 2^32
            # Consider IntegerTypes if a different type is consistently needed
            np.array(seq, dtype=np.uint32).tofile(os.path.join(processed_dir, f"{sp}.bin"))

        # Build meta.pkl content
        # Use HuggingFace Tokenizer's decode method
        tokenizer_for_meta = Tokenizer.from_file(str(tokenizer_path))
        # Create itos (ID to string) mapping using the new consecutive IDs
        itos = {new_id: tokenizer_for_meta.decode([old_id]) for old_id, new_id in old2new.items()}

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
        
        # Define a special EOT character for document separation (using a rarely used Unicode character)
        EOT_CHAR = '\x03'  # ETX (End of Text) control character
        
        # For multi-document mode, we need to include the EOT character in the vocabulary
        if is_multi_document:
            # Collect all unique characters from all documents plus EOT (parallel processing)
            # Only use multiprocessing when we have enough documents
            min_docs_for_parallel = max(actual_proc * 2, 10)
            if actual_proc > 1 and len(documents) >= min_docs_for_parallel:
                with Pool(actual_proc) as pool:
                    char_sets = pool.map(_get_unique_chars_from_doc, documents)
                all_chars = set().union(*char_sets)
            else:
                all_chars = set()
                for doc in documents:
                    all_chars.update(set(doc))
            all_chars.add(EOT_CHAR)
            chars = sorted(list(all_chars))
        else:
            # Single document mode: determine unique characters
            if actual_proc == 1:
                chars = sorted(list(get_unique_chars(data)))
            else:
                with Pool(actual_proc) as pool:
                    char_sets = pool.map(get_unique_chars, get_chunks(data, actual_proc))
                chars = sorted(list(set().union(*char_sets)))

        stoi = {ch: i for i, ch in enumerate(chars)} # Character to ID
        itos = {i: ch for ch, i in stoi.items()}    # ID to character
        vocab_size = len(chars)
        
        # Get EOT token ID (only valid for multi-document mode)
        eot_char_id = stoi.get(EOT_CHAR, None)
        
        # For multi-document mode with validation, we use hybrid stratified sampling:
        # - JSONL files: randomly shuffle documents, then split by document count
        # - TXT files: split by character ratio
        use_stratified_split_char = is_multi_document and not no_validation

        # Encode the dataset
        if use_stratified_split_char:
            # Stratified sampling mode: process each file independently based on its type
            train_encoded = []
            val_encoded = []
            
            # Process each file separately
            for file_docs, file_type in file_documents:
                if not file_docs:
                    continue
                
                # Encode all documents in this file
                min_docs_for_parallel = max(actual_proc * 2, 10)
                if actual_proc > 1 and len(file_docs) >= min_docs_for_parallel:
                    # Parallel encoding
                    args_list = [(doc, stoi, eot_char_id) for doc in file_docs]
                    with Pool(actual_proc) as pool:
                        results = pool.map(_encode_document_char, args_list)
                    file_token_lists = [(doc_tokens, eot_id) for doc_tokens, eot_id in results]
                else:
                    # Sequential encoding
                    file_token_lists = []
                    for doc in file_docs:
                        doc_tokens = [stoi.get(ch, 0) for ch in doc]
                        file_token_lists.append((doc_tokens, eot_char_id))
                
                if file_type == 'jsonl':
                    # JSONL files: randomly assign whole documents to train/val sets
                    indices = list(range(len(file_token_lists)))
                    random.shuffle(indices)
                    split_idx = int(len(indices) * train_split_ratio)
                    train_indices = set(indices[:split_idx])
                    
                    for i, (doc_tokens, eot_id) in enumerate(file_token_lists):
                        if not doc_tokens:
                            continue
                        doc_tokens_with_eot = list(doc_tokens)
                        if eot_id is not None:
                            doc_tokens_with_eot.append(eot_id)
                        
                        if i in train_indices:
                            train_encoded.extend(doc_tokens_with_eot)
                        else:
                            val_encoded.extend(doc_tokens_with_eot)
                else:
                    # TXT files: split each document by ratio
                    for doc_tokens, eot_id in file_token_lists:
                        if not doc_tokens:
                            continue
                        doc_tokens_with_eot = list(doc_tokens)
                        if eot_id is not None:
                            doc_tokens_with_eot.append(eot_id)
                        
                        # Split this document by ratio
                        split_at = int(len(doc_tokens_with_eot) * train_split_ratio)
                        train_encoded.extend(doc_tokens_with_eot[:split_at])
                        val_encoded.extend(doc_tokens_with_eot[split_at:])
            
            # Set train/val ids directly from stratified split
            train_ids = np.array(train_encoded, dtype=IntegerTypes)
            val_ids = np.array(val_encoded, dtype=IntegerTypes)
            
        elif is_multi_document:
            # Multi-document mode without validation: just concatenate all
            min_docs_for_parallel = max(actual_proc * 2, 10)
            if actual_proc > 1 and len(documents) >= min_docs_for_parallel:
                args_list = [(doc, stoi, eot_char_id) for doc in documents]
                with Pool(actual_proc) as pool:
                    results = pool.map(_encode_document_char, args_list)
                encoded = []
                for doc_tokens, eot_id in results:
                    encoded.extend(doc_tokens)
                    if eot_id is not None:
                        encoded.append(eot_id)
            else:
                encoded = []
                for doc in documents:
                    doc_encoded = [stoi.get(ch, 0) for ch in doc]
                    encoded.extend(doc_encoded)
                    if eot_char_id is not None:
                        encoded.append(eot_char_id)
            train_ids = np.array(encoded, dtype=IntegerTypes)
            val_ids = None
        else:
            # Single document mode: use chunking for large files
            if actual_proc == 1:
                encoded = encode_text_chunk(data, stoi)
            else:
                with Pool(actual_proc) as pool:
                    enc_chunks = pool.starmap(encode_text_chunk, [(c, stoi) for c in get_chunks(data, actual_proc)])
                encoded = [e for ck in enc_chunks for e in ck] # Flatten
            
            # Global split for single document
            if no_validation:
                train_ids = np.array(encoded, dtype=IntegerTypes)
                val_ids = None
            else:
                split_at = int(len(encoded) * train_split_ratio)
                train_ids = np.array(encoded[:split_at], dtype=IntegerTypes)
                val_ids = np.array(encoded[split_at:], dtype=IntegerTypes)

        # Handle no_validation for multi-document mode (remove old val.bin if exists)
        if no_validation:
            val_bin_path = os.path.join(processed_dir, "val.bin")
            if os.path.exists(val_bin_path):
                try:
                    os.remove(val_bin_path)
                    print(f"Removed old validation set file: {val_bin_path}")
                except OSError as e:
                    print(f"Warning: Could not remove old validation set file {val_bin_path}: {e}")

        # Save processed data
        train_ids.tofile(os.path.join(processed_dir, "train.bin"))
        if val_ids is not None:
            val_ids.tofile(os.path.join(processed_dir, "val.bin"))

        # Add tokenizer information to meta (include EOT token info for multi-document mode)
        meta = {
            "vocab_size": vocab_size,
            "itos": itos,
            "stoi": stoi,
            "tokenizer": tok_name,
            "eot_token_id": eot_char_id if is_multi_document else None  # EOT token ID for character-level
        }
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
