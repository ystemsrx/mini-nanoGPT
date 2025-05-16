import os, pickle, math, io
from contextlib import nullcontext

import numpy as np
import tiktoken

import torch._dynamo
torch._dynamo.config.suppress_errors = True

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from multiprocessing import Pool, cpu_count
from pathlib import Path

# Defalt Config
from config import DEFAULT_CONFIG, LANG_JSON, IntegerTypes
from db_manager import DBManager
dbm = DBManager()

def compose_model_dirs(model_name: str, model_id: int):
    """
    Generate model directories:
        · raw_data_dir   : ./data/{model_name}_{id}/raw/
        · processed_dir  : ./data/{model_name}_{id}/processed/
        · out_dir        : ./out/{model_name}_{id}/
    Return (raw_data_dir, processed_data_dir, out_dir)
    """
    folder = f"{model_name}_{model_id}"
    raw_data_dir = os.path.join("data", folder, "raw")
    processed_data_dir = os.path.join("data", folder, "processed")
    out_dir = os.path.join("out", folder)
    return raw_data_dir, processed_data_dir, out_dir

# ------------------------ GPT Model -----------------------------#

class GPTConfig:
    def __init__(
        self, 
        vocab_size, 
        block_size, 
        n_layer, 
        n_head, 
        n_embd, 
        dropout, 
        bias=False
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

class GPT(nn.Module):
    """
    A minimal GPT-like model consisting of an embedding layer and a linear layer. 
    This is primarily for demonstration purposes.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

    def forward(self, idx, targets=None):
        # idx shape: (batch, time)
        b, t = idx.size()
        # Convert token indices to embeddings
        token_emb = self.token_embedding_table(idx)
        # Project embeddings onto vocab space
        logits = self.lm_head(token_emb)

        loss = None
        if targets is not None:
            # Flatten logits/targets for cross entropy
            logits_view = logits.view(b * t, -1)
            targets_view = targets.view(b * t)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates tokens one at a time, appending each new token to 'idx'. 
        The process continues until 'max_new_tokens' tokens have been added.
        """
        for _ in range(max_new_tokens):
            if idx.size(1) == 0:
                raise ValueError("Input sequence is empty. Please provide at least one token.")
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Scale logits by temperature
            logits = logits[:, -1, :] / temperature
            # If top_k is set, zero out logits not in top_k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                top_value = v[:, -1].unsqueeze(-1)
                logits[logits < top_value] = -float('Inf')
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def crop_block_size(self, block_size):
        """
        Adjust the model's internal block_size if needed. 
        Useful when resuming training with a different context length.
        """
        self.config.block_size = block_size
        self.block_size = block_size

##############################################################################
# Optimizer Configuration
##############################################################################

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
    Creates and returns an AdamW optimizer for the model's parameters, 
    ignoring those that do not require gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer

##############################################################################
# Data Processing
##############################################################################

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
    Load local tokenizer.json and encode a chunk of text.
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
    - If "Use tokenizer" is checked, it will first attempt to use `tokenizer.json` in the root directory;
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

    raw_dir, processed_dir, _ = compose_model_dirs(model_name, model_id)
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    # -------- 1. 读取文本 -------- #
    data = input_text.strip()
    if not data and input_dir.strip():
        for fn in (f for f in os.listdir(input_dir) if f.endswith(".txt")):
            with open(os.path.join(input_dir, fn), "r", encoding="utf-8") as f:
                data += f.read()
    if not data:
        raise ValueError("It seems that you haven't provided any text. Please check your input.")

    with open(os.path.join(raw_dir, "merged_input.txt"), "w", encoding="utf-8") as f:
        f.write(data)

    # -------- 2. Tokenize & 切分 -------- #
    size_mb = len(data.encode("utf-8")) / 1024 / 1024
    actual_proc = min(num_proc, cpu_count()) if size_mb > 100 else 1

    # ================================
    # 2-A. 使用分词器 (tokenizer.json / GPT-2)
    # ================================
    if use_gpt2_tokenizer:
        tokenizer_path = Path.cwd() / "tokenizer.json"

        # ---- ① 根目录存在 tokenizer.json → 使用 HuggingFace Tokenizers ----
        if tokenizer_path.exists():
            try:
                from tokenizers import Tokenizer  # 提前检测依赖
            except ImportError as e:
                raise ImportError(
                    "Detected tokenizer.json, but the `tokenizers` library is not installed in the current environment:\n"
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

        # ---- ② No tokenizer.json → Fallback to GPT-2 (tiktoken) ----
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
            # For custom tokenizer.json，use HF Tokenizers decode
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

        meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
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
        "train_size": train_sz
    }
    if not no_validation:
        res["val_size"] = val_sz
    return res

##############################################################################
# Training: includes stop signal, DDP, logging, visualization, and checkpoints
##############################################################################

stop_signal = False

def stop_training():
    """
    Sets a global stop signal to True, allowing the training loop to break gracefully.
    """
    global stop_signal
    stop_signal = True

def train_model_generator(
    data_dir,
    out_dir,
    plot_interval=DEFAULT_CONFIG["training"]["plot_interval"],
    log_interval=DEFAULT_CONFIG["training"]["log_interval"],
    num_eval_seeds=DEFAULT_CONFIG["training"]["num_eval_seeds"],
    save_best_val_checkpoint=DEFAULT_CONFIG["training"]["save_best_val_checkpoint"],
    init_from=DEFAULT_CONFIG["training"]["init_from"],
    gradient_accumulation_steps=DEFAULT_CONFIG["training"]["gradient_accumulation_steps"],
    batch_size=DEFAULT_CONFIG["training"]["batch_size"],
    block_size=DEFAULT_CONFIG["training"]["block_size"],
    n_layer=DEFAULT_CONFIG["training"]["n_layer"],
    n_head=DEFAULT_CONFIG["training"]["n_head"],
    n_embd=DEFAULT_CONFIG["training"]["n_embd"],
    dropout=DEFAULT_CONFIG["training"]["dropout"],
    bias=DEFAULT_CONFIG["training"]["bias"],
    learning_rate=DEFAULT_CONFIG["training"]["learning_rate"],
    max_iters=DEFAULT_CONFIG["training"]["max_iters"],
    weight_decay=DEFAULT_CONFIG["training"]["weight_decay"],
    beta1=DEFAULT_CONFIG["training"]["beta1"],
    beta2=DEFAULT_CONFIG["training"]["beta2"],
    lr_scheduler_type=DEFAULT_CONFIG["training"]["lr_scheduler_type"],
    warmup_iters=DEFAULT_CONFIG["training"]["warmup_iters"],
    lr_decay_iters=DEFAULT_CONFIG["training"]["lr_decay_iters"],
    min_lr=DEFAULT_CONFIG["training"]["min_lr"],
    step_size=DEFAULT_CONFIG["training"]["step_size"],
    step_gamma=DEFAULT_CONFIG["training"]["step_gamma"],
    polynomial_power=DEFAULT_CONFIG["training"]["polynomial_power"],
    backend=DEFAULT_CONFIG["training"]["backend"],
    device=DEFAULT_CONFIG["training"]["device"],
    dtype=DEFAULT_CONFIG["training"]["dtype"],
    compile_model=DEFAULT_CONFIG["training"]["compile_model"],
    seed=DEFAULT_CONFIG["training"]["seed"],
    save_interval=DEFAULT_CONFIG["training"]["save_interval"]
):
    """
    Main training loop with optional validation set, DDP for distributed training,
    periodic logging, plotting, safe stop signals, checkpointing at best val loss,
    and the ability to run evaluation-only.

    -------------- Additional Features --------------
    · All training parameters and log paths are written to SQLite database (see db_manager.py)
    · Only relative paths are stored for easy migration
    """
    # ------------------- 0. Register/Save Training Config in Database ------------------- #
    # -- DB Integration --
    model_name = os.path.basename(os.path.abspath(out_dir)) or "new_model"
    model_id = dbm.register_model(model_name, out_dir)
    
    # Package all training hyperparameters into dict (for later UI restoration)
    _training_cfg_local_vars = dict(
        data_dir=data_dir,
        out_dir=out_dir,
        plot_interval=plot_interval,
        log_interval=log_interval,
        num_eval_seeds=num_eval_seeds,
        save_best_val_checkpoint=save_best_val_checkpoint,
        init_from=init_from,
        gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size,
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        learning_rate=learning_rate,
        max_iters=max_iters,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        lr_scheduler_type=lr_scheduler_type,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
        step_size=step_size,
        step_gamma=step_gamma,
        polynomial_power=polynomial_power,
        backend=backend,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
        seed=seed,
        save_interval=save_interval
    )
    # Write when first entered (subsequent entries will overwrite)
    dbm.save_training_config(model_id, _training_cfg_local_vars)
    # -------------------------------------------------------------------- #

    global stop_signal
    stop_signal = False

    def make_progress_html(progress_val, max_val, color='black'):
        html = (
            f"<div style='width: 100%; height: 20px; margin-bottom: 5px;'>"
            f"<progress value='{progress_val}' max='{max_val}' "
            f"style='width: 100%; height: 20px; color: {color};'></progress>"
            "</div>"
        )
        return html
    
    try:
        num_eval_seeds = int(num_eval_seeds)
        if num_eval_seeds < 0 or num_eval_seeds > 2**32 - 1:
            raise ValueError("Seed for evaluation must be between 0 and 2^32 - 1.")
    except ValueError as e:
        if num_eval_seeds != 0:
            error_msg = f"Error in evaluation seeds: {str(e)}"
            print(error_msg)
            yield (f"<div style='color: red;'>{error_msg}</div>", error_msg, None)
            return
        else:
            num_eval_seeds = 0

    if num_eval_seeds == 0:
        try:
            if not (0 <= seed <= 2**32 - 1):
                raise ValueError
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        except ValueError:
            msg = "Error: seed must be between 0 and 2^32 - 1."
            print(msg)
            yield (f"<div style='color: red;'>{msg}</div>", msg, None)
            return

    ddp = int(os.environ.get('RANK', -1)) != -1
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)
        master_process = (ddp_rank == 0)
        seed_offset = ddp_rank
        if num_eval_seeds == 0 and seed != 0:
            torch.manual_seed(seed + seed_offset)
            torch.cuda.manual_seed(seed + seed_offset)
            np.random.seed(seed + seed_offset)
        assert gradient_accumulation_steps % ddp_world_size == 0, \
            "gradient_accumulation_steps must be divisible by world size."
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        ddp_world_size = 1
        seed_offset = 0

    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        if num_eval_seeds == 0:
            print(f"Training starts, seed={seed} ...")
        else:
            print(f"Evaluation only, seeds={num_eval_seeds} ...")

    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Prepare logging
    loss_log_path = os.path.join(out_dir, 'loss_log.pkl')
    train_plot_steps = []
    train_plot_losses = []
    val_plot_steps = []
    val_plot_losses = []

    # Verify dataset existence
    train_bin_path = os.path.join(data_dir, 'train.bin')
    val_bin_path = os.path.join(data_dir, 'val.bin')
    has_val = os.path.exists(val_bin_path)

    if num_eval_seeds > 0 and not has_val:
        err = f"Error: val.bin not found, can't evaluate."
        print(err)
        yield (f"<div style='color:red;'>{err}</div>", err, None)
        return

    if not os.path.exists(train_bin_path) and num_eval_seeds == 0:
        err = f"Error: train.bin not found, can't train."
        print(err)
        yield (f"<div style='color:red;'>{err}</div>", err, None)
        return

    def get_batch(split="train"):
        train_data_memmap = np.memmap(train_bin_path, dtype=IntegerTypes, mode='r')
        if has_val:
            val_data_memmap = np.memmap(val_bin_path, dtype=IntegerTypes, mode='r')
            if len(train_data_memmap) <= len(val_data_memmap):
                min_data_memmap = len(train_data_memmap)
                min_dataset_name = "train"
            else:
                min_data_memmap = len(val_data_memmap)
                min_dataset_name = "val"
        else:
            min_data_memmap = len(train_data_memmap)
            min_dataset_name = "train"

        if split == 'train':
            data_memmap = train_data_memmap
        else:
            if not has_val:
                raise ValueError("No validation set.")
            data_memmap = val_data_memmap

        max_val_ = len(data_memmap) - block_size
        if max_val_ <= 0:
            raise ValueError(
                f"Dataset too small: minimum dataset({min_dataset_name}) size is {min_data_memmap}, "
                f"but block size is {block_size}. Either reduce block size or add more data."
            )

        ix = torch.randint(max_val_, (batch_size,))
        x = torch.stack([torch.from_numpy(data_memmap[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data_memmap[i+1:i+1+block_size].astype(np.int64)) for i in ix])

        if device_type == 'cuda':
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        err = f"Error: meta.pkl not found at {meta_path}"
        print(err)
        yield (f"<div style='color:red;'>{err}</div>", err, None)
        return
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']

    model_args = dict(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=bias,
        vocab_size=vocab_size,
        dropout=dropout
    )

    iter_num = 0
    best_val_loss = 1e9

    if num_eval_seeds > 0:
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    else:
        if init_from == 'scratch':
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
        elif init_from == 'resume':
            ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            if not os.path.exists(ckpt_path):
                msg = f"Error: cannot resume, {ckpt_path} not found."
                print(msg)
                yield (f"<div style='color:red;'>{msg}</div>", msg, None)
                return
            checkpoint = torch.load(ckpt_path, map_location=device)
            ckpt_args = checkpoint['model_args']
            for k, v in ckpt_args.items():
                if k in model_args:
                    model_args[k] = v
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            unwanted_prefix = '_orig_mod.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            iter_num = checkpoint['iter_num']
            best_val_loss = checkpoint['best_val_loss']

            loss_dict = None
            if os.path.exists(loss_log_path):
                with open(loss_log_path, 'rb') as f:
                    loss_dict = pickle.load(f)
                train_plot_steps = loss_dict.get('train_plot_steps', [])
                train_plot_losses = loss_dict.get('train_plot_losses', [])
                val_plot_steps = loss_dict.get('val_plot_steps', [])
                val_plot_losses = loss_dict.get('val_plot_losses', [])
        else:
            msg = "Error: please choose 'scratch' or 'resume'."
            print(msg)
            yield (f"<div style='color:red;'>{msg}</div>", msg, None)
            return

    if block_size < model_args['block_size']:
        model.crop_block_size(block_size)

    model.to(device)

    # ------------------------------------------------------------------------
    # EVALUATION-ONLY MODE
    # ------------------------------------------------------------------------
    if num_eval_seeds > 0:
        if not has_val:
            msg = f"Error: val.bin not found, can't evaluate."
            print(msg)
            yield (f"<div style='color:red;'>{msg}</div>", msg, None)
            return
        stoi, itos = meta['stoi'], meta['itos']
        model.eval()
        if compile_model:
            model = torch.compile(model)

        val_data_memmap = np.memmap(val_bin_path, dtype=IntegerTypes, mode='r')
        if block_size > len(val_data_memmap):
            msg = f"Error: block_size({block_size}) > validation set size({len(val_data_memmap)})."
            print(msg)
            yield (f"<div style='color:red;'>{msg}</div>", msg, None)
            return

        val_loss_list = []

        for seed_idx in range(1, num_eval_seeds + 1):
            if stop_signal:
                stop_msg = f"Evaluation stopped. Evaluated {seed_idx - 1} seeds."
                print(stop_msg)
                yield (make_progress_html(seed_idx - 1, num_eval_seeds, color='orange'), stop_msg, None)
                break
            current_seed = seed + seed_idx
            try:
                torch.manual_seed(current_seed)
                torch.cuda.manual_seed(current_seed)
                np.random.seed(current_seed)
            except ValueError as e:
                error_msg = f"Error: seed {current_seed} is invalid."
                print(error_msg)
                yield (make_progress_html(seed_idx, num_eval_seeds, color='orange'), error_msg, None)
                continue

            try:
                X_val, Y_val = get_batch('val')
            except ValueError as e:
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                yield (make_progress_html(seed_idx, num_eval_seeds, color='orange'), error_msg, None)
                break

            try:
                with ctx:
                    _, val_loss = model(X_val, Y_val)
                val_loss_val = val_loss.item()
            except Exception as e:
                val_loss_val = "Error"
                error_msg = f"Error: {str(e)}"
                print(error_msg)
                yield (make_progress_html(seed_idx, num_eval_seeds, color='orange'), error_msg, None)
                continue

            val_loss_list.append(val_loss_val if isinstance(val_loss_val, float) else 0.0)

            fig, ax = plt.subplots()
            ax.plot(
                range(1, len(val_loss_list) + 1),
                val_loss_list,
                label="Validation Loss",
                color='orange',
                marker='o'
            )
            ax.set_xlabel("Index")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.grid(True)
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=300)
            plt.close(fig)
            buf.seek(0)
            img_pil = Image.open(buf)

            if isinstance(val_loss_val, float):
                log_buffer = f"{seed_idx}. Seed: {current_seed}, val_loss={val_loss_val:.4f}"
            else:
                log_buffer = f"{seed_idx}. Seed: {current_seed}, val_loss=Error"

            print(log_buffer)
            progress_html = make_progress_html(seed_idx, num_eval_seeds, color='orange')
            yield (progress_html, log_buffer, img_pil)

        if master_process and not stop_signal:
            end_msg = f"Evaluation done. Seeds used: {num_eval_seeds}"
            print(end_msg)
            progress_html = make_progress_html(num_eval_seeds, num_eval_seeds, color='orange')
            if val_loss_list:
                fig, ax = plt.subplots()
                ax.plot(range(1, len(val_loss_list) + 1), val_loss_list, label="Validation Loss", color='orange', marker='o')
                ax.set_xlabel("Index")
                ax.set_ylabel("Loss")
                ax.legend()
                ax.grid(True)
                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=300)
                plt.close(fig)
                buf.seek(0)
                final_img_pil = Image.open(buf)
            else:
                final_img_pil = None
            yield (progress_html, end_msg, final_img_pil)
        return

    # ------------------------------------------------------------------------
    # TRAINING MODE
    # ------------------------------------------------------------------------
    if num_eval_seeds == 0:
        optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
        if init_from == 'resume':
            optimizer.load_state_dict(checkpoint['optimizer'])
        if compile_model:
            model = torch.compile(model)

        raw_model = model
        if ddp:
            ddp_local_rank = int(os.environ['LOCAL_RANK'])
            model = DDP(model, device_ids=[ddp_local_rank])
            raw_model = model.module
        scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

        def get_lr(it):
            if it < warmup_iters:
                return learning_rate * (it + 1) / (warmup_iters + 1)
            if lr_scheduler_type == "none":
                return learning_rate
            if lr_scheduler_type == "cosine":
                if it > lr_decay_iters:
                    return min_lr
                decay_ratio = (it - warmup_iters) / float(lr_decay_iters - warmup_iters)
                coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
                return min_lr + coeff * (learning_rate - min_lr)
            elif lr_scheduler_type == "constant_with_warmup":
                return learning_rate
            elif lr_scheduler_type == "linear":
                if it > lr_decay_iters:
                    return min_lr
                decay_ratio = (it - warmup_iters) / float(lr_decay_iters - warmup_iters)
                lr_ = learning_rate + (min_lr - learning_rate) * decay_ratio
                return lr_
            elif lr_scheduler_type == "step":
                effective_iter = max(0, it - warmup_iters)
                n_decay = effective_iter // step_size
                lr_ = learning_rate * (step_gamma ** n_decay)
                return max(lr_, min_lr)
            elif lr_scheduler_type == "polynomial":
                if it > lr_decay_iters:
                    return min_lr
                progress = float(it - warmup_iters) / float(lr_decay_iters - warmup_iters)
                poly = (1 - progress) ** polynomial_power
                lr_ = (learning_rate - min_lr) * poly + min_lr
                return lr_
            else:
                return learning_rate

        log_buffer = ""

    last_log = ""
    last_plot = None

    while True:
        if stop_signal:
            if master_process:
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss
                }
                final_ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(ckpt, final_ckpt_path)

                with open(loss_log_path, 'wb') as f:
                    pickle.dump({
                        'train_plot_steps': train_plot_steps,
                        'train_plot_losses': train_plot_losses,
                        'val_plot_steps': val_plot_steps,
                        'val_plot_losses': val_plot_losses
                    }, f)

                # -- DB Integration -- Save training log path.
                dbm.save_training_log(model_id, loss_log_path)

                fig, ax = plt.subplots()
                ax.plot(train_plot_steps, train_plot_losses, label="train_loss")
                if has_val and len(val_plot_losses) > 0:
                    ax.plot(val_plot_steps, val_plot_losses, label="val_loss")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                final_img = Image.open(buf)

                stop_msg = "Training stopped, checkpoint saved."
                print(stop_msg)
                progress_html = make_progress_html(iter_num, max_iters)
                yield (progress_html, stop_msg, final_img)
            break

        # Training batch
        try:
            X, Y = get_batch('train')
        except ValueError as e:
            msg = f"Error: {str(e)}"
            print(msg)
            if master_process:
                progress_html = make_progress_html(iter_num, max_iters)
                yield (progress_html, msg, None)
            break

        # Forward + backward pass
        with ctx:
            logits, loss = model(X, Y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        train_loss_val = loss.item()

        # Save intermediate checkpoints periodically
        if save_interval > 0 and (iter_num + 1) % save_interval == 0:
            save_path = os.path.join(out_dir, f'step_{iter_num + 1}', 'ckpt.pt')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ckpt = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss
            }
            torch.save(ckpt, save_path)
            if master_process:
                log_msg = f"Checkpoint saved at step {iter_num + 1}: {save_path}"
                print(log_msg)

        log_update = None
        plot_update = None

        # Log training info at intervals
        if (iter_num % log_interval == 0):
            log_buffer = f"Step {iter_num}: Train loss={train_loss_val:.4f}, LR={get_lr(iter_num):.6f}"
            print(log_buffer)
            last_log = log_buffer
            log_update = last_log

        # Plot at intervals (both training and validation loss if available)
        if (iter_num % plot_interval == 0):
            train_plot_steps.append(iter_num)
            train_plot_losses.append(train_loss_val)

            val_loss_val = None
            if has_val:
                try:
                    Xv, Yv = get_batch('val')
                    with ctx:
                        _, val_loss_ = model(Xv, Yv)
                    val_loss_val = val_loss_.item()
                except Exception as e:
                    val_loss_val = None
                    error_msg = f"Error while evaluating val loss: {str(e)}"
                    print(error_msg)

            if has_val and (val_loss_val is not None):
                val_plot_steps.append(iter_num)
                val_plot_losses.append(val_loss_val)
                if save_best_val_checkpoint and (val_loss_val < best_val_loss):
                    best_val_loss = val_loss_val
                    best_ckpt_path = os.path.join(out_dir, "best_checkpoint", "ckpt.pt")
                    os.makedirs(os.path.dirname(best_ckpt_path), exist_ok=True)
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss
                    }
                    torch.save(ckpt, best_ckpt_path)
                    print(f"New best val_loss={val_loss_val:.4f}, checkpoint saved at {best_ckpt_path}")

            if master_process:
                # Save logs for plotting
                to_save = {
                    'train_plot_steps': train_plot_steps,
                    'train_plot_losses': train_plot_losses,
                    'val_plot_steps': val_plot_steps,
                    'val_plot_losses': val_plot_losses
                }
                with open(loss_log_path, 'wb') as f:
                    pickle.dump(to_save, f)

                fig, ax = plt.subplots()
                ax.plot(train_plot_steps, train_plot_losses, label="train_loss")
                if has_val and len(val_plot_losses) > 0:
                    ax.plot(val_plot_steps, val_plot_losses, label="val_loss")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                img_pil = Image.open(buf)
                plot_update = img_pil

            # Yield latest logs and plot
            if log_update or plot_update:
                progress_html = make_progress_html(iter_num, max_iters)
                yield (progress_html, last_log, plot_update if plot_update else last_plot)
                if plot_update:
                    last_plot = plot_update

        # Update learning rate
        lr_now = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        iter_num += 1
        if iter_num > max_iters:
            if master_process:
                msg = f"Training finished: reached {max_iters} iterations."
                print(msg)
                ckpt = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss
                }
                torch.save(ckpt, os.path.join(out_dir, 'ckpt.pt'))

                # -- DB Integration -- 保存日志文件路径
                dbm.save_training_log(model_id, loss_log_path)

                fig, ax = plt.subplots()
                ax.plot(train_plot_steps, train_plot_losses, label="train_loss")
                if has_val and len(val_plot_losses) > 0:
                    ax.plot(val_plot_steps, val_plot_losses, label="val_loss")
                ax.set_xlabel("Iteration")
                ax.set_ylabel("Loss")
                ax.legend()
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                final_img = Image.open(buf)

                yield (make_progress_html(iter_num, max_iters), msg, final_img)
            break

    if ddp:
        destroy_process_group()

    return

# -------------- Inference: loads from out_dir/ckpt.pt------------------

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
    Generates text from a single checkpoint and把推理配置/历史写入数据库。
    若 out_dir 以 .pt 结尾，则直接当作 ckpt 路径；否则默认 out_dir/ckpt.pt。
    """
    # ---------------- 0. Database: ensure model_id & record config ---------------- #
    # -- DB Integration --
    ckpt_dir = out_dir if out_dir.endswith('.pt') else os.path.join(out_dir, 'ckpt.pt')
    model_dir_for_db = os.path.dirname(ckpt_dir)  # 用目录定位模型
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
    # ------------------------------------------------------------------ #

    if not prompt.strip():
        yield "Prompt is empty, please provide a starting text."
        return

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # ----------- 1. Analyze ckpt path -----------
        ckpt_path = ckpt_dir
        if not os.path.exists(ckpt_path):
            err = f"Error: checkpoint not found at {ckpt_path}."
            yield err
            return

        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        if compile_model:
            model = torch.compile(model)

        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            yield f"Error: meta.pkl not found at {meta_path}."
            return

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        stoi, itos = meta['stoi'], meta['itos']

        def encode(s):
            return [stoi.get(ch, 0) for ch in s]
        def decode(l):
            return ''.join([itos.get(i, '') for i in l])

        xids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
        block_size = gptconf.block_size
        if xids.size(1) > block_size:
            yield f"Error: input length ({xids.size(1)}) exceeds block size ({block_size})."
            return

        # ----------- 3. Generate text & accumulate output ------------
        accumulated_output = []
        with torch.no_grad():
            with ctx:
                for s_i in range(num_samples):
                    idx = xids.clone()
                    generated = prompt
                    for _ in range(max_new_tokens):
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

                        generated_tokens = idx[0].tolist()
                        generated = decode(generated_tokens)

                    sample_txt = f"Sample {s_i+1}:\n{generated}"
                    accumulated_output.append(sample_txt)
                    yield sample_txt
                    if s_i < num_samples - 1:
                        yield "-" * 20

        final_text = "\n\n".join(accumulated_output)

        # ---------------- 3. Write inference history to DB ------------------ #
        # -- DB Integration --
        dbm.save_inference_history(model_id, final_text)

    except Exception as ex:
        yield f"An unexpected error occurred: {str(ex)}"

# --------------- Building the Gradio App ---------------------

def build_app_interface(selected_lang: str = "zh"):
    """
        Top-level UI function
        Implemented:
          · Logic for new model/model name, automatic directory, dropdown refresh/delete  
          · Language switching: After switching the `lang_select` dropdown, **all component labels & default values** are refreshed synchronously  
    """

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def _model_choices():
        return [f"{m['id']} - {m['name']}" for m in dbm.get_all_models()]

    def _load_loss_plot(loss_log_path: str):
        if not (loss_log_path and os.path.exists(loss_log_path)):
            return None
        try:
            with open(loss_log_path, "rb") as f:
                loss_dict = pickle.load(f)
            tr_steps = loss_dict.get("train_plot_steps", [])
            tr_losses = loss_dict.get("train_plot_losses", [])
            val_steps = loss_dict.get("val_plot_steps", [])
            val_losses = loss_dict.get("val_plot_losses", [])
            if not tr_steps:
                return None
            fig, ax = plt.subplots()
            ax.plot(tr_steps, tr_losses, label="train")
            if val_losses:
                ax.plot(val_steps, val_losses, label="val")
            ax.set_xlabel("Iter"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)
            buf = io.BytesIO(); plt.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
            return Image.open(buf)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Initialize Gradio app
    # ------------------------------------------------------------------ #
    T = LANG_JSON[selected_lang]

    custom_css = """
    .gradio-container{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,Cantarell,'Open Sans','Helvetica Neue',sans-serif;}
    progress{width:100%;height:20px;margin:4px 0;}
    #train-log-box{height:150px;overflow-y:auto;font-family:monospace;padding:8px;background:white;}
    """

    # ------------------------------------------------------------------ #
    # —— Blocks
    # ------------------------------------------------------------------ #
    with gr.Blocks(title=T["app_title"], css=custom_css) as demo:

        # ========= Top: model management / language ========= #
        with gr.Row():
            model_dropdown     = gr.Dropdown(label=T["registered_models"], choices=_model_choices(), interactive=True)
            refresh_models_btn = gr.Button(T["refresh_tables"])
            delete_model_btn   = gr.Button(T["delete_selected_model"], variant="stop")

        lang_select = gr.Dropdown(label=T["language_label"],
                                  choices=list(LANG_JSON.keys()),
                                  value=selected_lang, interactive=True)

        # ========= Tabs ========= #
        with gr.Tabs() as main_tabs:

            # -------------- Data processing Tab -------------- #
            with gr.Tab(T["data_process_tab"]) as data_process_tab:
                with gr.Row():
                    input_text = gr.Textbox(label=T["dp_paste_text"], lines=19.5)
                    with gr.Column():
                        txt_dir        = gr.Textbox(label=T["dp_txt_dir"], value="")
                        new_model_chk  = gr.Checkbox(label=T["new_model"], value=True)
                        model_name_box = gr.Textbox(label=T["model_name"], value="new_model")
                        with gr.Row():
                            no_val_set = gr.Checkbox(label=T["dp_no_val_set"],
                                                     value=DEFAULT_CONFIG["data_process"]["no_validation"])
                            use_gpt2   = gr.Checkbox(label=T["dp_use_gpt2_tokenizer"],
                                                     value=DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"])
                        train_split = gr.Slider(label=T["dp_train_split"],
                                                minimum=0.1, maximum=0.99, step=0.01,
                                                value=DEFAULT_CONFIG["data_process"]["train_split_ratio"])
                        num_proc    = gr.Number(label=T["dp_num_proc"],
                                                value=DEFAULT_CONFIG["data_process"]["num_proc"],
                                                precision=0)
                process_btn    = gr.Button(T["dp_start_btn"])
                process_output = gr.Textbox(label=T["dp_result"], lines=5, interactive=False)

            # -------------- Training Tab -------------- #
            with gr.Tab(T["train_tab"]) as train_tab:
                train_params_title_md = gr.Markdown(f"### {T['train_params_title']}")

                with gr.Row():
                    data_dir_box = gr.Textbox(label=T["train_data_dir"], value="", interactive=False)
                    out_dir_box  = gr.Textbox(label=T["train_out_dir"],  value="", interactive=False)
                    backend_box  = gr.Textbox(label=T["train_backend"],  value=DEFAULT_CONFIG["training"]["backend"])
                    device_box   = gr.Dropdown(label=T["train_device"],  choices=["cpu","cuda"],
                                               value=DEFAULT_CONFIG["training"]["device"])
                    dtype_box    = gr.Dropdown(label=T["train_dtype"],   choices=["float16","bfloat16","float32"],
                                               value=DEFAULT_CONFIG["training"]["dtype"])
                    compile_box  = gr.Checkbox(label=T["train_compile_model"],
                                               value=DEFAULT_CONFIG["training"]["compile_model"])

                with gr.Row():
                    plot_interval_box       = gr.Number(label=T["train_eval_interval"],
                                                        value=DEFAULT_CONFIG["training"]["plot_interval"])
                    log_interval_box        = gr.Number(label=T["train_log_interval"],
                                                        value=DEFAULT_CONFIG["training"]["log_interval"])
                    num_eval_seeds_box      = gr.Number(label=T["train_num_eval_seeds"],
                                                        value=DEFAULT_CONFIG["training"]["num_eval_seeds"])
                    save_best_val_ckpt_box  = gr.Checkbox(label=T["train_save_best_val_ckpt"],
                                                          value=DEFAULT_CONFIG["training"]["save_best_val_checkpoint"])
                    init_from_box           = gr.Dropdown(label=T["train_init_from"],
                                                          choices=["scratch","resume"],
                                                          value=DEFAULT_CONFIG["training"]["init_from"])
                    seed_box                = gr.Number(label=T["train_seed"],
                                                        value=DEFAULT_CONFIG["training"]["seed"])

                with gr.Row():
                    grad_acc_box  = gr.Number(label=T["train_gas"],
                                              value=DEFAULT_CONFIG["training"]["gradient_accumulation_steps"])
                    batch_size_box = gr.Number(label=T["train_batch_size"],
                                               value=DEFAULT_CONFIG["training"]["batch_size"])
                    block_size_box = gr.Number(label=T["train_block_size"],
                                               value=DEFAULT_CONFIG["training"]["block_size"])
                    n_layer_box    = gr.Number(label=T["train_n_layer"],
                                               value=DEFAULT_CONFIG["training"]["n_layer"])
                    n_head_box     = gr.Number(label=T["train_n_head"],
                                               value=DEFAULT_CONFIG["training"]["n_head"])
                    n_embd_box     = gr.Number(label=T["train_n_embd"],
                                               value=DEFAULT_CONFIG["training"]["n_embd"])

                with gr.Row():
                    dropout_box      = gr.Number(label=T["train_dropout"],
                                                 value=DEFAULT_CONFIG["training"]["dropout"])
                    bias_box         = gr.Checkbox(label=T["train_bias"],
                                                   value=DEFAULT_CONFIG["training"]["bias"])
                    lr_box           = gr.Number(label=T["train_lr"],
                                                 value=DEFAULT_CONFIG["training"]["learning_rate"])
                    max_iters_box    = gr.Number(label=T["train_max_iters"],
                                                 value=DEFAULT_CONFIG["training"]["max_iters"])
                    weight_decay_box = gr.Number(label=T["train_weight_decay"],
                                                 value=DEFAULT_CONFIG["training"]["weight_decay"])

                with gr.Row():
                    beta1_box        = gr.Number(label=T["train_beta1"],
                                                 value=DEFAULT_CONFIG["training"]["beta1"])
                    beta2_box        = gr.Number(label=T["train_beta2"],
                                                 value=DEFAULT_CONFIG["training"]["beta2"])
                    lr_scheduler_box = gr.Dropdown(label=T["train_lr_scheduler"],
                                                   choices=["none","cosine","constant_with_warmup",
                                                            "linear","step","polynomial"],
                                                   value=DEFAULT_CONFIG["training"]["lr_scheduler_type"])
                    warmup_box       = gr.Number(label=T["train_warmup_iters"],
                                                 value=DEFAULT_CONFIG["training"]["warmup_iters"])
                    lr_decay_box     = gr.Number(label=T["train_lr_decay_iters"],
                                                 value=DEFAULT_CONFIG["training"]["lr_decay_iters"])
                    min_lr_box       = gr.Number(label=T["train_min_lr"],
                                                 value=DEFAULT_CONFIG["training"]["min_lr"])

                with gr.Row():
                    step_size_box        = gr.Number(label="Step Size",
                                                     value=DEFAULT_CONFIG["training"]["step_size"])
                    step_gamma_box       = gr.Number(label="Step Gamma",
                                                     value=DEFAULT_CONFIG["training"]["step_gamma"])
                    polynomial_power_box = gr.Number(label="Polynomial Power",
                                                     value=DEFAULT_CONFIG["training"]["polynomial_power"])
                    save_interval_box    = gr.Number(label=T["train_save_interval"],
                                                     value=DEFAULT_CONFIG["training"]["save_interval"])

                train_btn = gr.Button(T["train_start_btn"])
                stop_btn  = gr.Button(T["stop_btn"])

                with gr.Row():
                    with gr.Column(scale=1):
                        train_progress = gr.HTML(label="Training Progress")
                        train_log      = gr.Textbox(label=T["train_log"],
                                                    elem_id="train-log-box", interactive=False)
                    with gr.Column(scale=2):
                        train_plot     = gr.Image(label=T["train_plot"], type="pil")

            # -------------- 推理 Tab -------------- #
            with gr.Tab(T["infer_tab"]) as inf_tab:
                with gr.Row():
                    data_dir_inf = gr.Textbox(label=T["dp_processed_dir"], value="", interactive=False)
                    out_dir_inf  = gr.Textbox(label=T["inf_out_dir"], value="", interactive=False)

                prompt_box = gr.Textbox(label=T["inf_prompt"],
                                        value=DEFAULT_CONFIG["inference"]["prompt"], lines=5)

                with gr.Row():
                    num_samples_box    = gr.Number(label=T["inf_num_samples"],
                                                   value=DEFAULT_CONFIG["inference"]["num_samples"])
                    max_new_tokens_box = gr.Number(label=T["inf_max_new_tokens"],
                                                   value=DEFAULT_CONFIG["inference"]["max_new_tokens"])
                    temperature_box    = gr.Number(label=T["inf_temperature"],
                                                   value=DEFAULT_CONFIG["inference"]["temperature"])
                    top_k_box          = gr.Number(label=T["inf_top_k"],
                                                   value=DEFAULT_CONFIG["inference"]["top_k"])
                    seed_box_inf       = gr.Number(label=T["inf_seed"],
                                                   value=DEFAULT_CONFIG["inference"]["seed"])

                inf_btn    = gr.Button(T["inf_start_btn"])
                inf_output = gr.Textbox(label=T["inf_result"], lines=10, interactive=False)

        # ------------------------------------------------------------------ #
        # Call backs: data processing / training / inference
        # ------------------------------------------------------------------ #
        def data_processing_cb(
            new_flag, model_name, dropdown_val,
            txt, ddir,
            sp, no_val, use_gpt2_tokenizer, num_proc_
        ):
            try:
                # Get current language
                current_lang = lang_select.value
                T_current = LANG_JSON[current_lang]
                
                sel_id = int(dropdown_val.split(" - ")[0]) if dropdown_val and " - " in dropdown_val else None
                info = process_data(
                    model_name=model_name.strip() or "unnamed",
                    new_model=new_flag,
                    selected_model_id=sel_id,
                    input_text=txt,
                    input_dir=ddir,
                    train_split_ratio=sp,
                    no_validation=no_val,
                    use_gpt2_tokenizer=use_gpt2_tokenizer,
                    num_proc=int(num_proc_)
                )
                new_choices = _model_choices()
                new_val     = f"{info['model_id']} - {model_name.strip() or 'unnamed'}"
                msg = (
                    f"✅ {T_current['dp_result']}:\n"  # Use current langeuage
                    f"model_id = {info['model_id']}\n"
                    f"processed_dir = {info['processed_data_dir']}\n"
                    f"vocab_size = {info['vocab_size']}\n"
                    f"train_size = {info['train_size']}" +
                    (f"\nval_size = {info['val_size']}" if 'val_size' in info else "\n(no val)")
                )
                return msg, gr.update(choices=new_choices, value=new_val)
            except Exception as e:
                return f"❌ Error: {str(e)}", gr.update()

        process_btn.click(
            fn=data_processing_cb,
            inputs=[new_model_chk, model_name_box, model_dropdown,
                    input_text, txt_dir,
                    train_split, no_val_set, use_gpt2, num_proc],
            outputs=[process_output, model_dropdown]
        )

        # ------------------------------------------------------------------ #
        # Call backs: stop training
        # ------------------------------------------------------------------ #
        stop_btn.click(fn=stop_training, inputs=[], outputs=[])

        # ------------------------------------------------------------------ #
        # Call backs: start training
        # ------------------------------------------------------------------ #
        def training_cb(
            data_dir_, out_dir_, plot_interval_, log_interval_, num_eval_seeds_,
            save_best_val_ckpt_, init_from_,
            grad_acc_, batch_size_, block_size_,
            n_layer_, n_head_, n_embd_,
            dropout_, bias_,
            lr_, max_iters_, weight_decay_,
            beta1_, beta2_,
            lr_scheduler_type_, warmup_,
            lr_decay_, min_lr_,
            step_size_, step_gamma_, polynomial_power_,
            backend_, device_, dtype_, compile_,
            seed_, save_interval_
        ):
            img_pil = None
            try:
                num_eval_seeds_int = int(num_eval_seeds_)
                if num_eval_seeds_int < 0 or num_eval_seeds_int > 2**32-1:
                    raise ValueError("seed out of range")
            except ValueError as e:
                yield (f"<div style='color:red;'>{str(e)}</div>", str(e), img_pil); return

            try:
                gen = train_model_generator(
                    data_dir=data_dir_,
                    out_dir=out_dir_,
                    plot_interval=int(plot_interval_),
                    log_interval=int(log_interval_),
                    num_eval_seeds=int(num_eval_seeds_),
                    save_best_val_checkpoint=bool(save_best_val_ckpt_),
                    init_from=init_from_,
                    gradient_accumulation_steps=int(grad_acc_),
                    batch_size=int(batch_size_), block_size=int(block_size_),
                    n_layer=int(n_layer_), n_head=int(n_head_), n_embd=int(n_embd_),
                    dropout=float(dropout_), bias=bool(bias_),
                    learning_rate=float(lr_), max_iters=int(max_iters_),
                    weight_decay=float(weight_decay_),
                    beta1=float(beta1_), beta2=float(beta2_),
                    lr_scheduler_type=lr_scheduler_type_,
                    warmup_iters=int(warmup_), lr_decay_iters=int(lr_decay_),
                    min_lr=float(min_lr_), step_size=int(step_size_),
                    step_gamma=float(step_gamma_), polynomial_power=float(polynomial_power_),
                    backend=backend_, device=device_, dtype=dtype_,
                    compile_model=bool(compile_), seed=int(seed_), save_interval=int(save_interval_)
                )
                for p_html, log_html, img in gen:
                    yield (p_html, log_html, img)
            except Exception as e:
                err = f"Error: {str(e)}"
                yield (f"<div style='color:red;'>{err}</div>", err, img_pil)

        train_btn.click(
            fn=training_cb,
            inputs=[
                data_dir_box, out_dir_box,
                plot_interval_box, log_interval_box, num_eval_seeds_box,
                save_best_val_ckpt_box, init_from_box,
                grad_acc_box, batch_size_box, block_size_box,
                n_layer_box, n_head_box, n_embd_box,
                dropout_box, bias_box,
                lr_box, max_iters_box, weight_decay_box,
                beta1_box, beta2_box,
                lr_scheduler_box, warmup_box,
                lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box,
                backend_box, device_box, dtype_box, compile_box,
                seed_box, save_interval_box
            ],
            outputs=[train_progress, train_log, train_plot]
        )

        # ------------------------------------------------------------------ #
        # Call backs: inference
        # ------------------------------------------------------------------ #
        def inference_cb(
            data_dir_inf_, out_dir_inf_,
            prompt_, num_samples_, max_new_tokens_,
            temperature_, top_k_, seed_inf_
        ):
            try:
                gen = generate_text(
                    data_dir=data_dir_inf_, out_dir=out_dir_inf_,
                    prompt=prompt_,
                    num_samples=int(num_samples_),
                    max_new_tokens=int(max_new_tokens_),
                    temperature=float(temperature_),
                    top_k=int(top_k_) if top_k_ else None,
                    seed=int(seed_inf_),
                    device=DEFAULT_CONFIG["inference"]["device"],
                    dtype=DEFAULT_CONFIG["inference"]["dtype"],
                    compile_model=DEFAULT_CONFIG["inference"]["compile_model"]
                )
                acc = ""
                for piece in gen:
                    acc += piece + "\n\n"
                    yield acc.strip()
            except Exception as e:
                yield f"Error: {str(e)}"

        inf_btn.click(
            fn=inference_cb,
            inputs=[data_dir_inf, out_dir_inf, prompt_box,
                    num_samples_box, max_new_tokens_box,
                    temperature_box, top_k_box, seed_box_inf],
            outputs=inf_output
        )

        # ------------------------------------------------------------------ #
        # Call backs: model selection
        # ------------------------------------------------------------------ #
        def _reset_updates():
            def _d(val=""): return gr.update(value=val)
            d = DEFAULT_CONFIG
            return [
                gr.update(value=True), _d("new_model"),          # new_model_chk & model_name_box
                _d(), _d(),                                      # data_dir_box, out_dir_box
                _d(d["training"]["plot_interval"]), _d(d["training"]["log_interval"]),
                _d(d["training"]["num_eval_seeds"]),
                _d(d["training"]["save_best_val_checkpoint"]),
                _d(d["training"]["init_from"]),
                _d(d["training"]["gradient_accumulation_steps"]),
                _d(d["training"]["batch_size"]),
                _d(d["training"]["block_size"]),
                _d(d["training"]["n_layer"]),
                _d(d["training"]["n_head"]),
                _d(d["training"]["n_embd"]),
                _d(d["training"]["dropout"]),
                _d(d["training"]["bias"]),
                _d(d["training"]["learning_rate"]),
                _d(d["training"]["max_iters"]),
                _d(d["training"]["weight_decay"]),
                _d(d["training"]["beta1"]),
                _d(d["training"]["beta2"]),
                _d(d["training"]["lr_scheduler_type"]),
                _d(d["training"]["warmup_iters"]),
                _d(d["training"]["lr_decay_iters"]),
                _d(d["training"]["min_lr"]),
                _d(d["training"]["step_size"]),
                _d(d["training"]["step_gamma"]),
                _d(d["training"]["polynomial_power"]),
                _d(d["training"]["backend"]),
                _d(d["training"]["device"]),
                _d(d["training"]["dtype"]),
                _d(d["training"]["compile_model"]),
                _d(d["training"]["seed"]),
                _d(d["training"]["save_interval"]),
                None, "",                                     # train_plot, train_log
                _d(), _d(),                                   # data_dir_inf, out_dir_inf
                _d(d["inference"]["prompt"]),
                _d(d["inference"]["num_samples"]),
                _d(d["inference"]["max_new_tokens"]),
                _d(d["inference"]["temperature"]),
                _d(d["inference"]["top_k"]),
                _d(d["inference"]["seed"]),
                ""                                            # inf_output
            ]

        def select_model_cb(sel: str):
            if not sel:
                return _reset_updates()

            try:
                mid = int(sel.split(" - ")[0])
            except ValueError:
                return _reset_updates()

            cfg  = dbm.get_training_config(mid)  or {}
            icfg = dbm.get_inference_config(mid) or {}
            info = dbm.get_model_basic_info(mid) or {}
            name = info.get("name", "")
            folder = f"{name}_{mid}"
            data_processed_dir = os.path.join("data", folder, "processed")
            out_dir_root       = os.path.join("out", folder)

            def _cfg(k, d):  return cfg.get(k, d)
            def _ic(k, d):   return icfg.get(k, d)

            loss_plot   = _load_loss_plot(dbm.get_training_log_path(mid))
            train_log_s = ""

            updates = [
                gr.update(value=False),     # new_model_chk
                gr.update(value=name),      # model_name_box
                gr.update(value=data_processed_dir),   # data_dir_box
                gr.update(value=out_dir_root),         # out_dir_box
                gr.update(value=_cfg("plot_interval",           DEFAULT_CONFIG["training"]["plot_interval"])),
                gr.update(value=_cfg("log_interval",            DEFAULT_CONFIG["training"]["log_interval"])),
                gr.update(value=_cfg("num_eval_seeds",          DEFAULT_CONFIG["training"]["num_eval_seeds"])),
                gr.update(value=_cfg("save_best_val_checkpoint",DEFAULT_CONFIG["training"]["save_best_val_checkpoint"])),
                gr.update(value=_cfg("init_from",               DEFAULT_CONFIG["training"]["init_from"])),
                gr.update(value=_cfg("gradient_accumulation_steps",
                                     DEFAULT_CONFIG["training"]["gradient_accumulation_steps"])),
                gr.update(value=_cfg("batch_size",      DEFAULT_CONFIG["training"]["batch_size"])),
                gr.update(value=_cfg("block_size",      DEFAULT_CONFIG["training"]["block_size"])),
                gr.update(value=_cfg("n_layer",         DEFAULT_CONFIG["training"]["n_layer"])),
                gr.update(value=_cfg("n_head",          DEFAULT_CONFIG["training"]["n_head"])),
                gr.update(value=_cfg("n_embd",          DEFAULT_CONFIG["training"]["n_embd"])),
                gr.update(value=_cfg("dropout",         DEFAULT_CONFIG["training"]["dropout"])),
                gr.update(value=_cfg("bias",            DEFAULT_CONFIG["training"]["bias"])),
                gr.update(value=_cfg("learning_rate",   DEFAULT_CONFIG["training"]["learning_rate"])),
                gr.update(value=_cfg("max_iters",       DEFAULT_CONFIG["training"]["max_iters"])),
                gr.update(value=_cfg("weight_decay",    DEFAULT_CONFIG["training"]["weight_decay"])),
                gr.update(value=_cfg("beta1",           DEFAULT_CONFIG["training"]["beta1"])),
                gr.update(value=_cfg("beta2",           DEFAULT_CONFIG["training"]["beta2"])),
                gr.update(value=_cfg("lr_scheduler_type", DEFAULT_CONFIG["training"]["lr_scheduler_type"])),
                gr.update(value=_cfg("warmup_iters",    DEFAULT_CONFIG["training"]["warmup_iters"])),
                gr.update(value=_cfg("lr_decay_iters",  DEFAULT_CONFIG["training"]["lr_decay_iters"])),
                gr.update(value=_cfg("min_lr",          DEFAULT_CONFIG["training"]["min_lr"])),
                gr.update(value=_cfg("step_size",       DEFAULT_CONFIG["training"]["step_size"])),
                gr.update(value=_cfg("step_gamma",      DEFAULT_CONFIG["training"]["step_gamma"])),
                gr.update(value=_cfg("polynomial_power",DEFAULT_CONFIG["training"]["polynomial_power"])),
                gr.update(value=_cfg("backend",         DEFAULT_CONFIG["training"]["backend"])),
                gr.update(value=_cfg("device",          DEFAULT_CONFIG["training"]["device"])),
                gr.update(value=_cfg("dtype",           DEFAULT_CONFIG["training"]["dtype"])),
                gr.update(value=_cfg("compile_model",   DEFAULT_CONFIG["training"]["compile_model"])),
                gr.update(value=_cfg("seed",            DEFAULT_CONFIG["training"]["seed"])),
                gr.update(value=_cfg("save_interval",   DEFAULT_CONFIG["training"]["save_interval"])),
                loss_plot, train_log_s,
                gr.update(value=data_processed_dir),    # data_dir_inf
                gr.update(value=out_dir_root),          # out_dir_inf
                gr.update(value=_ic("prompt",         DEFAULT_CONFIG["inference"]["prompt"])),
                gr.update(value=_ic("num_samples",    DEFAULT_CONFIG["inference"]["num_samples"])),
                gr.update(value=_ic("max_new_tokens", DEFAULT_CONFIG["inference"]["max_new_tokens"])),
                gr.update(value=_ic("temperature",    DEFAULT_CONFIG["inference"]["temperature"])),
                gr.update(value=_ic("top_k",          DEFAULT_CONFIG["inference"]["top_k"])),
                gr.update(value=_ic("seed",           DEFAULT_CONFIG["inference"]["seed"])),
                dbm.get_inference_history(mid) or ""
            ]
            return updates

        model_dropdown.change(
            fn=select_model_cb,
            inputs=[model_dropdown],
            outputs=[
                new_model_chk, model_name_box,
                data_dir_box, out_dir_box,
                plot_interval_box, log_interval_box,
                num_eval_seeds_box, save_best_val_ckpt_box, init_from_box,
                grad_acc_box, batch_size_box, block_size_box,
                n_layer_box, n_head_box, n_embd_box,
                dropout_box, bias_box,
                lr_box, max_iters_box, weight_decay_box,
                beta1_box, beta2_box,
                lr_scheduler_box, warmup_box,
                lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box,
                backend_box, device_box, dtype_box, compile_box,
                seed_box, save_interval_box,
                train_plot, train_log,
                data_dir_inf, out_dir_inf,
                prompt_box, num_samples_box, max_new_tokens_box,
                temperature_box, top_k_box, seed_box_inf,
                inf_output
            ]
        )

        # ------------------------------------------------------------------ #
        # Call backs: delete model
        # ------------------------------------------------------------------ #
        def delete_model_cb(sel: str):
            if sel and " - " in sel:
                try:
                    dbm.delete_model(int(sel.split(" - ")[0]))
                except Exception:
                    pass
                    
            new_choices = _model_choices()
            return gr.update(choices=new_choices, value=None), *_reset_updates()

        delete_model_btn.click(
            fn=delete_model_cb,
            inputs=[model_dropdown],
            outputs=[model_dropdown,
                     new_model_chk, model_name_box,
                     data_dir_box, out_dir_box,
                     plot_interval_box, log_interval_box,
                     num_eval_seeds_box, save_best_val_ckpt_box, init_from_box,
                     grad_acc_box, batch_size_box, block_size_box,
                     n_layer_box, n_head_box, n_embd_box,
                     dropout_box, bias_box,
                     lr_box, max_iters_box, weight_decay_box,
                     beta1_box, beta2_box,
                     lr_scheduler_box, warmup_box,
                     lr_decay_box, min_lr_box,
                     step_size_box, step_gamma_box, polynomial_power_box,
                     backend_box, device_box, dtype_box, compile_box,
                     seed_box, save_interval_box,
                     train_plot, train_log,
                     data_dir_inf, out_dir_inf,
                     prompt_box, num_samples_box, max_new_tokens_box,
                     temperature_box, top_k_box, seed_box_inf,
                     inf_output]
        )

        refresh_models_btn.click(lambda: gr.update(choices=_model_choices()), [], [model_dropdown])

        # ------------------------------------------------------------------ #
        # Call backs: language switch
        # ------------------------------------------------------------------ #
        def switch_language(lang_code: str):
            Tn = LANG_JSON[lang_code]
            return [
                gr.update(label=Tn["language_label"], value=lang_code),
                # Tab labels
                gr.update(label=Tn["data_process_tab"]),
                gr.update(label=Tn["train_tab"]),
                gr.update(label=Tn["infer_tab"]),
        
                # Top bar
                gr.update(label=Tn["registered_models"]),  # model_dropdown
                gr.update(value=Tn["refresh_tables"]),     # Button: refresh_models_btn
                gr.update(value=Tn["delete_selected_model"]), # Button: delete_model_btn
        
                # Model management panel
                gr.update(label=Tn["new_model"]),        # new_model_chk
                gr.update(label=Tn["model_name"]),       # model_name_box
        
                # Data processing panel
                gr.update(label=Tn["dp_paste_text"]),      # input_text
                gr.update(label=Tn["dp_txt_dir"]),         # txt_dir
                gr.update(label=Tn["dp_no_val_set"]),      # no_val_set
                gr.update(label=Tn["dp_use_gpt2_tokenizer"]), # use_gpt2
                gr.update(label=Tn["dp_train_split"]),     # train_split
                gr.update(label=Tn["dp_num_proc"]),        # num_proc
                gr.update(value=Tn["dp_start_btn"]),       # process_btn (Button)
                gr.update(label=Tn["dp_result"]),          # process_output
                
                # Training panel
                gr.update(value=f"### {Tn['train_params_title']}"), # train_params_title_md
                gr.update(label=Tn["train_data_dir"]),     # data_dir_box
                gr.update(label=Tn["train_out_dir"]),      # out_dir_box
                gr.update(label=Tn["train_backend"]),      # backend_box
                gr.update(label=Tn["train_device"]),       # device_box
                gr.update(label=Tn["train_dtype"]),        # dtype_box
                gr.update(label=Tn["train_compile_model"]), # compile_box
                gr.update(label=Tn["train_eval_interval"]), # plot_interval_box
                gr.update(label=Tn["train_log_interval"]), # log_interval_box
                gr.update(label=Tn["train_num_eval_seeds"]), # num_eval_seeds_box
                gr.update(label=Tn["train_save_best_val_ckpt"]), # save_best_val_ckpt_box
                gr.update(label=Tn["train_init_from"]),    # init_from_box
                gr.update(label=Tn["train_seed"]),         # seed_box
                gr.update(label=Tn["train_gas"]),          # grad_acc_box
                gr.update(label=Tn["train_batch_size"]),   # batch_size_box
                gr.update(label=Tn["train_block_size"]),   # block_size_box
                gr.update(label=Tn["train_n_layer"]),      # n_layer_box
                gr.update(label=Tn["train_n_head"]),       # n_head_box
                gr.update(label=Tn["train_n_embd"]),       # n_embd_box
                gr.update(label=Tn["train_dropout"]),      # dropout_box
                gr.update(label=Tn["train_bias"]),         # bias_box
                gr.update(label=Tn["train_lr"]),           # lr_box
                gr.update(label=Tn["train_max_iters"]),    # max_iters_box
                gr.update(label=Tn["train_weight_decay"]), # weight_decay_box
                gr.update(label=Tn["train_beta1"]),        # beta1_box
                gr.update(label=Tn["train_beta2"]),        # beta2_box
                gr.update(label=Tn["train_lr_scheduler"]), # lr_scheduler_box
                gr.update(label=Tn["train_warmup_iters"]), # warmup_box
                gr.update(label=Tn["train_lr_decay_iters"]), # lr_decay_box
                gr.update(label=Tn["train_min_lr"]),       # min_lr_box
                gr.update(label=Tn["train_save_interval"]), # save_interval_box
                gr.update(value=Tn["train_start_btn"]),    # train_btn (Button)
                gr.update(value=Tn["stop_btn"]),           # stop_btn (Button)
                gr.update(label=Tn["train_log"]),          # train_log
                gr.update(label=Tn["train_plot"]),         # train_plot
                
                # Inference panel
                gr.update(label=Tn["dp_processed_dir"]),   # data_dir_inf
                gr.update(label=Tn["inf_out_dir"]),        # out_dir_inf
                gr.update(label=Tn["inf_prompt"]),         # prompt_box
                gr.update(label=Tn["inf_num_samples"]),    # num_samples_box
                gr.update(label=Tn["inf_max_new_tokens"]), # max_new_tokens_box
                gr.update(label=Tn["inf_temperature"]),    # temperature_box
                gr.update(label=Tn["inf_top_k"]),          # top_k_box
                gr.update(value=Tn["inf_start_btn"]),      # inf_btn (Button)
                gr.update(label=Tn["inf_result"]),         # inf_output
                gr.update(label=Tn["inf_seed"])            # seed_box_inf
            ]

        lang_select.change(
            fn=switch_language,
            inputs=[lang_select],
            outputs=[
                lang_select,
                data_process_tab, train_tab, inf_tab,
                model_dropdown, refresh_models_btn, delete_model_btn,
                new_model_chk, model_name_box,
                input_text, txt_dir,
                no_val_set, use_gpt2,
                train_split, num_proc, process_btn, process_output,
                train_params_title_md,
                data_dir_box, out_dir_box,
                backend_box, device_box, dtype_box, compile_box,
                plot_interval_box, log_interval_box, num_eval_seeds_box,
                save_best_val_ckpt_box, init_from_box, seed_box,
                grad_acc_box, batch_size_box, block_size_box,
                n_layer_box, n_head_box, n_embd_box,
                dropout_box, bias_box,
                lr_box, max_iters_box, weight_decay_box,
                beta1_box, beta2_box, lr_scheduler_box,
                warmup_box, lr_decay_box, min_lr_box,
                save_interval_box, train_btn, stop_btn,
                train_log, train_plot,
                data_dir_inf, out_dir_inf, prompt_box,
                num_samples_box, max_new_tokens_box,
                temperature_box, top_k_box, inf_btn, inf_output,
                seed_box_inf
            ]
        )

    return demo

# ----------------- Launch -------------------
if __name__=="__main__":
    demo = build_app_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)

