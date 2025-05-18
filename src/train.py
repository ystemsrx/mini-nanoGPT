# src/train.py

import os
import pickle
import math
import io
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import DEFAULT_CONFIG, IntegerTypes
from src.db_manager import DBManager
from src.gpt_model import GPTConfig, GPT, configure_optimizers

dbm = DBManager()

# Global stop signal for graceful interruption
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
