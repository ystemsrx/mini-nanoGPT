# src/train.py

import os
import pickle
import math
# import io # No longer needed for image buffer
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F # Keep for model or other utilities
from torch.optim import AdamW # Keep for optimizer
from torch.nn.parallel import DistributedDataParallel as DDP # Keep for DDP
from torch.distributed import init_process_group, destroy_process_group # Keep for DDP
# from PIL import Image # No longer needed for generating plots in this file

# import matplotlib # No longer needed
# matplotlib.use('Agg') # No longer needed
# import matplotlib.pyplot as plt # No longer needed

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
    # ... (0. Register/Save Training Config in Database - unchanged) ...
    model_name = os.path.basename(os.path.abspath(out_dir)) or "new_model" # Ensure out_dir is absolute first
    # If out_dir might be relative, ensure it's handled, e.g. model_name = os.path.basename(out_dir)
    # The original code has `os.path.basename(os.path.abspath(out_dir))`. Let's assume out_dir is valid path.
    
    # Robust model_name extraction
    abs_out_dir = os.path.abspath(out_dir)
    model_name_from_path = os.path.basename(abs_out_dir)
    if not model_name_from_path and abs_out_dir.endswith(os.path.sep): # Handle trailing slash case
        model_name_from_path = os.path.basename(os.path.dirname(abs_out_dir))
    
    model_name = model_name_from_path or "new_model" # Fallback if path is unusual
    model_id = dbm.register_model(model_name, out_dir)
    
    _training_cfg_local_vars = dict( # All params passed to the function
        data_dir=data_dir, out_dir=out_dir, plot_interval=plot_interval, log_interval=log_interval,
        num_eval_seeds=num_eval_seeds, save_best_val_checkpoint=save_best_val_checkpoint,
        init_from=init_from, gradient_accumulation_steps=gradient_accumulation_steps,
        batch_size=batch_size, block_size=block_size, n_layer=n_layer, n_head=n_head, n_embd=n_embd,
        dropout=dropout, bias=bias, learning_rate=learning_rate, max_iters=max_iters,
        weight_decay=weight_decay, beta1=beta1, beta2=beta2, lr_scheduler_type=lr_scheduler_type,
        warmup_iters=warmup_iters, lr_decay_iters=lr_decay_iters, min_lr=min_lr,
        step_size=step_size, step_gamma=step_gamma, polynomial_power=polynomial_power,
        backend=backend, device=device, dtype=dtype, compile_model=compile_model,
        seed=seed, save_interval=save_interval
    )
    dbm.save_training_config(model_id, _training_cfg_local_vars)

    global stop_signal
    stop_signal = False

    def make_progress_html(progress_val, max_val, color='black'):
        # Ensure max_val is not zero to avoid issues if max_iters is 0 (though unlikely)
        # The progress element handles value > max by capping at max.
        html = (
            f"<div style='width: 100%; height: 20px; margin-bottom: 5px;'>"
            f"<progress value='{progress_val}' max='{max_val if max_val > 0 else 1}' " # Ensure max_val > 0
            f"style='width: 100%; height: 20px; color: {color};'></progress>"
            "</div>"
        )
        return html
    
    # Initial empty plot data tuple
    empty_plot_data = ([], [], [], [])

    try:
        # ... (Seed validation, DDP setup, device setup - unchanged) ...
        num_eval_seeds = int(num_eval_seeds)
        if num_eval_seeds < 0 or num_eval_seeds > 2**32 - 1: # unsigned 32-bit int range
            raise ValueError("Seed for evaluation must be between 0 and 2^32 - 1.")
    except ValueError as e:
        if num_eval_seeds != 0: # Only error if num_eval_seeds was non-zero and invalid
            error_msg = f"Error in evaluation seeds: {str(e)}"
            print(error_msg)
            yield (f"<div style='color: red;'>{error_msg}</div>", error_msg, empty_plot_data) # Yield empty data
            return
        else: # num_eval_seeds was 0 (training mode), this error path shouldn't be hit for num_eval_seeds itself
            num_eval_seeds = 0 # Explicitly ensure it's 0 if it was an invalid string for 0

    # Main seed validation (for training or if num_eval_seeds is 0)
    try:
        current_seed_val = int(seed)
        if not (0 <= current_seed_val <= 2**32 - 1):
            raise ValueError("Seed must be between 0 and 2^32 - 1.")
        # Set seed if not DDP or if master process (DDP handles offset)
    except ValueError as e:
        msg = f"Error: seed '{seed}' is invalid. {str(e)}"
        print(msg)
        yield (f"<div style='color: red;'>{msg}</div>", msg, empty_plot_data)
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
        if num_eval_seeds == 0 : # Apply seed offset for training in DDP
             torch.manual_seed(current_seed_val + seed_offset)
             torch.cuda.manual_seed(current_seed_val + seed_offset)
             np.random.seed(current_seed_val + seed_offset)
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        master_process = True
        ddp_world_size = 1 # For consistency
        seed_offset = 0
        if num_eval_seeds == 0: # Apply seed for non-DDP training
            torch.manual_seed(current_seed_val)
            torch.cuda.manual_seed(current_seed_val)
            np.random.seed(current_seed_val)


    if master_process:
        os.makedirs(out_dir, exist_ok=True)
        if num_eval_seeds == 0:
            print(f"Training starts, seed={current_seed_val} ...")
        else:
            print(f"Evaluation only, num_eval_seeds={num_eval_seeds}, base_seed={current_seed_val} ...")


    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    loss_log_path = os.path.join(out_dir, 'loss_log.pkl')
    train_plot_steps, train_plot_losses = [], []
    val_plot_steps, val_plot_losses = [], []
    current_plot_data = (train_plot_steps, train_plot_losses, val_plot_steps, val_plot_losses) # Persist for non-plot yields

    # ... (Dataset verification, get_batch, meta loading - unchanged) ...
    train_bin_path = os.path.join(data_dir, 'train.bin')
    val_bin_path = os.path.join(data_dir, 'val.bin')
    has_val = os.path.exists(val_bin_path)

    # Check for train.bin if in training mode (num_eval_seeds == 0)
    if num_eval_seeds == 0 and not os.path.exists(train_bin_path):
        err = f"Error: train.bin not found at {train_bin_path}, can't train."
        print(err)
        yield (f"<div style='color:red;'>{err}</div>", err, empty_plot_data)
        return

    # Check for val.bin if validation is expected (num_eval_seeds > 0 OR (num_eval_seeds == 0 AND has_val for periodic eval))
    if num_eval_seeds > 0 and not has_val: # Eval mode requires val.bin
        err = f"Error: val.bin not found at {val_bin_path}, can't evaluate."
        print(err)
        yield (f"<div style='color:red;'>{err}</div>", err, empty_plot_data)
        return
    
    # get_batch definition (ensure it uses the correct IntegerTypes from config)
    def get_batch(split="train"):
        # Ensure data files exist before trying to memmap
        current_data_path = train_bin_path if split == 'train' else val_bin_path
        if not os.path.exists(current_data_path):
            raise FileNotFoundError(f"{split}.bin not found at {current_data_path}")

        data_memmap = np.memmap(current_data_path, dtype=IntegerTypes, mode='r')
        
        # Check if dataset is smaller than block_size
        if len(data_memmap) < block_size +1: # Need at least block_size for x and 1 for y
             raise ValueError(
                f"Dataset '{split}' (size {len(data_memmap)}) is too small for block_size ({block_size})."
            )
        
        max_idx = len(data_memmap) - block_size
        ix = torch.randint(max_idx, (batch_size,)) # Corrected: max_idx should be exclusive upper bound for randint
        
        x_list = [torch.from_numpy(data_memmap[i:i+block_size].astype(np.int64)) for i in ix]
        y_list = [torch.from_numpy(data_memmap[i+1:i+1+block_size].astype(np.int64)) for i in ix]

        if not x_list or not y_list: # Should not happen if max_idx check is correct
            raise ValueError(f"Could not generate batch from {split} dataset. Indices: {ix}, max_idx: {max_idx}")

        x = torch.stack(x_list)
        y = torch.stack(y_list)

        if device_type == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


    meta_path = os.path.join(data_dir, 'meta.pkl')
    if not os.path.exists(meta_path):
        err = f"Error: meta.pkl not found at {meta_path}"
        print(err)
        yield (f"<div style='color:red;'>{err}</div>", err, empty_plot_data)
        return
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    vocab_size = meta['vocab_size']


    # ... (Model initialization, iter_num, best_val_loss - unchanged) ...
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size, bias=bias, vocab_size=vocab_size, dropout=dropout)
    iter_num = 0
    best_val_loss = 1e9 # Initialize with a large value

    if num_eval_seeds > 0: # Evaluation mode
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        # In eval mode, we need to load a checkpoint. Assume ckpt.pt in out_dir.
        ckpt_path_eval = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path_eval):
            msg = f"Error: Checkpoint {ckpt_path_eval} not found for evaluation mode."
            print(msg)
            yield (f"<div style='color:red;'>{msg}</div>", msg, empty_plot_data)
            return
        checkpoint = torch.load(ckpt_path_eval, map_location=device)
        # Optionally update model_args from checkpoint if they differ, though usually for eval, current args are used with loaded weights.
        # For simplicity, assuming the loaded checkpoint is compatible with current model_args or has its own.
        # The original code did this for 'resume' in training. For pure eval, it might be simpler.
        # Let's load the state dict.
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k_sd, v_sd in list(state_dict.items()):
            if k_sd.startswith(unwanted_prefix):
                state_dict[k_sd[len(unwanted_prefix):]] = state_dict.pop(k_sd)
        model.load_state_dict(state_dict)
        if 'iter_num' in checkpoint: iter_num = checkpoint['iter_num'] # For context
        if 'best_val_loss' in checkpoint: best_val_loss = checkpoint['best_val_loss']


    elif init_from == 'scratch': # Training mode from scratch
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
    elif init_from == 'resume': # Training mode, resume
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if not os.path.exists(ckpt_path):
            msg = f"Error: Cannot resume, checkpoint {ckpt_path} not found."
            print(msg)
            yield (f"<div style='color:red;'>{msg}</div>", msg, empty_plot_data)
            return
        checkpoint = torch.load(ckpt_path, map_location=device)
        # Update model_args from checkpoint to ensure consistency
        # ckpt_model_args = checkpoint['model_args'] # Using this is safer
        # for k_arg, v_arg in ckpt_model_args.items(): # Ensure all necessary args are present
        #    model_args[k_arg] = v_arg # This was in original, good for consistency
        
        # Re-checking original logic for model_args update:
        # It iterates over ckpt_args, not model_args. This is correct.
        ckpt_args_from_file = checkpoint['model_args']
        for k_check, v_check in ckpt_args_from_file.items():
            if k_check in model_args: # Update only if key exists in current model_args
                model_args[k_check] = v_check
            # else: print(f"Warning: Checkpoint arg '{k_check}' not in current model_args.") # Optional warning

        gptconf = GPTConfig(**model_args) # Create config with potentially updated args
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k_sd, v_sd in list(state_dict.items()): # Iterate over a copy of items for modification
            if k_sd.startswith(unwanted_prefix):
                state_dict[k_sd[len(unwanted_prefix):]] = state_dict.pop(k_sd)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        # Load previous plot data
        if os.path.exists(loss_log_path):
            with open(loss_log_path, 'rb') as f:
                loss_data_loaded = pickle.load(f)
            train_plot_steps = loss_data_loaded.get('train_plot_steps', [])
            train_plot_losses = loss_data_loaded.get('train_plot_losses', [])
            val_plot_steps = loss_data_loaded.get('val_plot_steps', [])
            val_plot_losses = loss_data_loaded.get('val_plot_losses', [])
            current_plot_data = (train_plot_steps, train_plot_losses, val_plot_steps, val_plot_losses)

    else: # Should not happen if UI restricts init_from
        msg = "Error: Invalid 'init_from' value. Choose 'scratch' or 'resume'."
        print(msg)
        yield (f"<div style='color:red;'>{msg}</div>", msg, empty_plot_data)
        return

    if block_size < model.config.block_size: # model.config.block_size is the original/loaded one
        model.crop_block_size(block_size) # Crop if current block_size is smaller
    elif block_size > model.config.block_size:
        # This case needs careful handling - model wasn't trained for larger block_size.
        # Usually, block_size from config is used. If user specifies a larger one, it might error.
        # For simplicity, assume block_size from UI is authoritative for new/scratch.
        # For resume, model_args block_size would have been updated from checkpoint if different.
        pass # If current block_size (from UI) is larger, the model's internal block_size (from config) remains.

    model.to(device)


    # ------------------------------------------------------------------------
    # EVALUATION-ONLY MODE
    # ------------------------------------------------------------------------
    if num_eval_seeds > 0:
        if not has_val: # Should have been caught earlier, but double-check
            msg = f"Error: val.bin not found, critical for evaluation mode."
            print(msg)
            yield (f"<div style='color:red;'>{msg}</div>", msg, empty_plot_data)
            return
        
        model.eval()
        if compile_model:
            print("Compiling model for evaluation...")
            try:
                model = torch.compile(model) # Potentially time-consuming
            except Exception as e_compile:
                print(f"Warning: Model compilation failed for evaluation: {e_compile}")


        all_eval_losses = []
        eval_steps_axis = [] # X-axis for this plot will be seed index

        for seed_idx_loop in range(num_eval_seeds): # Loop 0 to num_eval_seeds-1
            actual_seed_idx_display = seed_idx_loop + 1 # For display 1 to num_eval_seeds
            if stop_signal:
                stop_msg = f"Evaluation stopped by user. Evaluated {seed_idx_loop} seeds."
                print(stop_msg)
                # Yield current plot data for eval
                plot_data_for_eval = ([], [], eval_steps_axis[:], all_eval_losses[:])
                yield (make_progress_html(seed_idx_loop, num_eval_seeds, color='orange'), stop_msg, plot_data_for_eval)
                break
            
            current_eval_seed = current_seed_val + actual_seed_idx_display # Use base seed + offset
            # print(f"Evaluating with seed: {current_eval_seed}") # Debug
            torch.manual_seed(current_eval_seed)
            torch.cuda.manual_seed(current_eval_seed) # Handles non-cuda case too
            np.random.seed(current_eval_seed)

            try:
                X_val_batch, Y_val_batch = get_batch('val')
            except Exception as e_batch: # Catch FileNotFoundError or ValueError from get_batch
                error_msg = f"Error getting validation batch (seed {current_eval_seed}): {str(e_batch)}"
                print(error_msg)
                yield (make_progress_html(actual_seed_idx_display, num_eval_seeds, color='red'), error_msg, empty_plot_data)
                continue # Try next seed or stop if this is critical

            try:
                with torch.no_grad(): # Ensure no gradients are computed
                    with ctx:
                        _, current_val_loss_tensor = model(X_val_batch, Y_val_batch)
                current_val_loss_float = current_val_loss_tensor.item()
                all_eval_losses.append(current_val_loss_float)
                eval_steps_axis.append(actual_seed_idx_display)
                log_buffer_line = f"{actual_seed_idx_display}. Seed: {current_eval_seed}, val_loss={current_val_loss_float:.4f}"
            except Exception as e_eval:
                log_buffer_line = f"{actual_seed_idx_display}. Seed: {current_eval_seed}, val_loss=ERROR ({str(e_eval)})"
                # all_eval_losses.append(float('nan')) # Or skip appending if error
                # eval_steps_axis.append(actual_seed_idx_display)
                print(f"Error during model evaluation (seed {current_eval_seed}): {e_eval}")
            
            print(log_buffer_line)
            progress_html_eval = make_progress_html(actual_seed_idx_display, num_eval_seeds, color='orange')
            current_eval_plot_data = ([], [], eval_steps_axis[:], all_eval_losses[:]) # Train empty, Val has eval data
            yield (progress_html_eval, log_buffer_line, current_eval_plot_data)

        if master_process and not stop_signal: # Final yield for evaluation
            end_msg = f"Evaluation complete. Seeds evaluated: {len(all_eval_losses)}/{num_eval_seeds}."
            if all_eval_losses: # Calculate average if any losses were recorded
                avg_loss = np.mean([l for l in all_eval_losses if not np.isnan(l)] ) # filter out potential NaNs
                end_msg += f" Average Val Loss: {avg_loss:.4f}"
            print(end_msg)
            final_progress_html = make_progress_html(len(all_eval_losses), num_eval_seeds, color='green' if len(all_eval_losses) == num_eval_seeds else 'orange')
            final_eval_plot_data = ([], [], eval_steps_axis[:], all_eval_losses[:])
            yield (final_progress_html, end_msg, final_eval_plot_data)
        return # End of evaluation-only mode

    # ------------------------------------------------------------------------
    # TRAINING MODE (num_eval_seeds == 0)
    # ------------------------------------------------------------------------
    optimizer = configure_optimizers(model, weight_decay, learning_rate, (beta1, beta2), device_type)
    if init_from == 'resume' and 'optimizer' in checkpoint: # Check if optimizer state exists
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e_optim_load:
            print(f"Warning: Could not load optimizer state: {e_optim_load}. Continuing with fresh optimizer.")
    
    if compile_model:
        print("Compiling model for training...")
        try:
            model = torch.compile(model)
        except Exception as e_compile_train:
            print(f"Warning: Model compilation failed for training: {e_compile_train}")


    raw_model = model # Keep a reference to the unwrapped model
    if ddp:
        # ddp_local_rank is already defined if ddp is True
        model = DDP(model, device_ids=[ddp_local_rank])
        raw_model = model.module # Get the original model from DDP wrapper
    
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    def get_lr(it): # Learning rate scheduler logic - unchanged
        # ... (get_lr function as in original) ...
        if it < warmup_iters:
            return learning_rate * (it + 1) / (warmup_iters + 1) # Ensure warmup_iters > 0 if used
        
        # Handle lr_decay_iters being 0 or less than warmup_iters for certain schedulers
        effective_lr_decay_iters = max(lr_decay_iters, warmup_iters + 1) # Ensure decay phase has some length

        if lr_scheduler_type == "none":
            return learning_rate
        if lr_scheduler_type == "cosine":
            if it >= effective_lr_decay_iters: return min_lr
            decay_ratio = (it - warmup_iters) / float(effective_lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return min_lr + coeff * (learning_rate - min_lr)
        elif lr_scheduler_type == "constant_with_warmup":
            return learning_rate
        elif lr_scheduler_type == "linear":
            if it >= effective_lr_decay_iters: return min_lr
            decay_ratio = (it - warmup_iters) / float(effective_lr_decay_iters - warmup_iters)
            return learning_rate + (min_lr - learning_rate) * decay_ratio
        elif lr_scheduler_type == "step":
            if step_size <= 0: return learning_rate # Avoid division by zero or infinite loop
            effective_iter_step = max(0, it - warmup_iters)
            n_decay = effective_iter_step // step_size
            return max(learning_rate * (step_gamma ** n_decay), min_lr)
        elif lr_scheduler_type == "polynomial":
            if it >= effective_lr_decay_iters: return min_lr
            progress = float(it - warmup_iters) / float(effective_lr_decay_iters - warmup_iters)
            poly = (1.0 - progress) ** polynomial_power
            return (learning_rate - min_lr) * poly + min_lr
        else: # Fallback or unknown scheduler
            return learning_rate


    # Training loop
    last_log_message = "" # Store the last log message for yielding
    
    # Initial yield with current state (e.g., if resuming)
    if master_process and iter_num > 0 and init_from == 'resume':
        initial_progress_html = make_progress_html(iter_num, max_iters)
        initial_log = f"Resumed at step {iter_num}. Best val loss: {best_val_loss:.4f}"
        print(initial_log)
        yield (initial_progress_html, initial_log, current_plot_data)


    while True:
        if stop_signal:
            if master_process:
                # Save final checkpoint and logs
                ckpt = {
                    'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss
                }
                final_ckpt_path = os.path.join(out_dir, 'ckpt.pt')
                torch.save(ckpt, final_ckpt_path)
                print(f"Checkpoint saved to {final_ckpt_path} due to stop signal.")

                with open(loss_log_path, 'wb') as f:
                    pickle.dump({
                        'train_plot_steps': train_plot_steps, 'train_plot_losses': train_plot_losses,
                        'val_plot_steps': val_plot_steps, 'val_plot_losses': val_plot_losses
                    }, f)
                dbm.save_training_log(model_id, loss_log_path)
                
                stop_msg = "Training stopped by user, checkpoint saved."
                print(stop_msg)
                progress_html_stop = make_progress_html(iter_num, max_iters)
                # Yield final plot data
                final_plot_data_on_stop = (train_plot_steps[:], train_plot_losses[:], val_plot_steps[:], val_plot_losses[:])
                yield (progress_html_stop, stop_msg, final_plot_data_on_stop)
            break # Exit while loop

        # --- Training Batch and Forward/Backward Pass ---
        try:
            X, Y = get_batch('train')
        except Exception as e_get_batch_train: # Catch FileNotFoundError or ValueError
            msg = f"Error getting training batch: {str(e_get_batch_train)}"
            print(msg)
            if master_process:
                yield (make_progress_html(iter_num, max_iters), msg, current_plot_data)
            break # Critical error, stop training

        # Forward + backward pass
        # Standard DDP gradient accumulation loop
        for micro_step in range(gradient_accumulation_steps * ddp_world_size): # Multiply back by ddp_world_size if it was divided
            if ddp: # DDP sync logic for gradient accumulation
                # Adjust sync context based on whether it's the last micro_step
                # This logic is complex and depends on how grad_acc_steps was handled with DDP.
                # Assuming model.no_sync() for all but the last micro_step if accumulation is > 1
                # The original code divides grad_acc_steps by ddp_world_size, implying each rank does fewer acc steps.
                # For simplicity, let's assume the accumulation loop is handled per rank.
                # The provided code implies `gradient_accumulation_steps` is per-rank after division.
                # So, the loop `range(gradient_accumulation_steps)` is correct per rank.
                # Synchronization happens naturally at `loss.backward()` or `scaler.step()`.
                 pass # DDP handles grad sync implicitly unless explicitly managed with no_sync

            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps # Scale loss for accumulation

            # Immediately async prefetch next batch while model is doing forwards
            # (This is an optimization, can be omitted for simplicity if causing issues)
            # X_next, Y_next = get_batch('train') # If prefetching

            scaler.scale(loss).backward()
            # X, Y = X_next, Y_next # If prefetching

        # Clip gradients (optional but good practice)
        # grad_clip = 1.0 # Example value
        # scaler.unscale_(optimizer) # Unscale before clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        train_loss_val = loss.item() * gradient_accumulation_steps # Rescale for logging

        # Save intermediate checkpoints periodically
        if save_interval > 0 and (iter_num + 1) % save_interval == 0 and master_process:
            save_dir_step = os.path.join(out_dir, f'step_{iter_num + 1}')
            os.makedirs(save_dir_step, exist_ok=True)
            save_path_step = os.path.join(save_dir_step, 'ckpt.pt')
            ckpt_step = {
                'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss
            }
            torch.save(ckpt_step, save_path_step)
            print(f"Intermediate checkpoint saved at step {iter_num + 1}: {save_path_step}")


        # --- Logging and Plotting ---
        new_log_generated = False
        new_plot_data_generated = False

        if master_process: # Only master process handles logging, plotting, and yielding updates
            if iter_num % log_interval == 0:
                lr_current_log = get_lr(iter_num)
                last_log_message = f"Step {iter_num}: Train loss={train_loss_val:.4f}, LR={lr_current_log:.6f}"
                if has_val and val_plot_losses: # Add last val loss if available
                    last_log_message += f", Last Val Loss={val_plot_losses[-1]:.4f} (at step {val_plot_steps[-1]})"
                print(last_log_message)
                new_log_generated = True

            if iter_num % plot_interval == 0:
                train_plot_steps.append(iter_num)
                train_plot_losses.append(train_loss_val)
                
                current_val_loss_for_plot = None
                if has_val:
                    model.eval() # Switch to eval mode for validation
                    try:
                        Xv, Yv = get_batch('val')
                        with torch.no_grad(): # Ensure no gradients during validation
                            with ctx:
                                _, val_loss_tensor = model(Xv, Yv)
                        current_val_loss_for_plot = val_loss_tensor.item()
                        val_plot_steps.append(iter_num)
                        val_plot_losses.append(current_val_loss_for_plot)

                        if save_best_val_checkpoint and (current_val_loss_for_plot < best_val_loss):
                            best_val_loss = current_val_loss_for_plot
                            best_ckpt_dir = os.path.join(out_dir, "best_checkpoint")
                            os.makedirs(best_ckpt_dir, exist_ok=True)
                            best_ckpt_path = os.path.join(best_ckpt_dir, "ckpt.pt")
                            ckpt_best = {
                                'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss
                            }
                            torch.save(ckpt_best, best_ckpt_path)
                            print(f"New best val_loss={best_val_loss:.4f} at step {iter_num}, checkpoint saved.")
                    except Exception as e_val:
                        print(f"Error during validation step {iter_num}: {str(e_val)}")
                    model.train() # Switch back to train mode

                # Update persistent plot data
                current_plot_data = (train_plot_steps[:], train_plot_losses[:], val_plot_steps[:], val_plot_losses[:])
                new_plot_data_generated = True
                
                # Save loss log to pickle file
                with open(loss_log_path, 'wb') as f:
                    pickle.dump({
                        'train_plot_steps': train_plot_steps, 'train_plot_losses': train_plot_losses,
                        'val_plot_steps': val_plot_steps, 'val_plot_losses': val_plot_losses
                    }, f)
                dbm.save_training_log(model_id, loss_log_path) # Update DB with log path

            # Yield update if new log or new plot data
            if new_log_generated or new_plot_data_generated:
                progress_html_train = make_progress_html(iter_num, max_iters)
                # Yield the most current log and the overall current plot data
                yield (progress_html_train, last_log_message, current_plot_data)
        
        # Update learning rate
        lr_update = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_update

        iter_num += 1
        if iter_num > max_iters:
            if master_process:
                msg_max_iters = f"Training finished: reached {max_iters} iterations."
                print(msg_max_iters)
                # Save final checkpoint
                ckpt_final = {
                    'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'model_args': model_args, 'iter_num': iter_num, 'best_val_loss': best_val_loss
                }
                torch.save(ckpt_final, os.path.join(out_dir, 'ckpt.pt'))
                # Ensure loss log is saved one last time
                with open(loss_log_path, 'wb') as f:
                    pickle.dump({
                        'train_plot_steps': train_plot_steps, 'train_plot_losses': train_plot_losses,
                        'val_plot_steps': val_plot_steps, 'val_plot_losses': val_plot_losses
                    }, f)
                dbm.save_training_log(model_id, loss_log_path)

                final_plot_data_max_iters = (train_plot_steps[:], train_plot_losses[:], val_plot_steps[:], val_plot_losses[:])
                yield (make_progress_html(iter_num, max_iters), msg_max_iters, final_plot_data_max_iters)
            break # Exit while loop

    if ddp:
        destroy_process_group()

    # The generator implicitly returns/stops after the loop or break