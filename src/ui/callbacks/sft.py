import os

import torch
import gradio as gr

from src.config import DEFAULT_CONFIG, LANG_JSON
from src.sft import validate_alpaca_format, load_sft_dataset, sft_train_generator, stop_sft_training
from src.infer_cache import ModelCache
from src.ui.charts import generate_loss_chart_html, make_progress_html
from src.ui.state import dbm


def update_sft_lr_scheduler_params(scheduler_type, cfg=None):
    defaults_sft = DEFAULT_CONFIG["sft"]

    def _val(key):
        if cfg and key in cfg:
            return cfg[key]
        return defaults_sft[key]

    # Use None for disabled Number components to avoid type errors
    warmup_update = gr.update(interactive=False, value=None)
    lr_decay_update = gr.update(interactive=False, value=None)
    min_lr_update = gr.update(interactive=False, value=None)
    step_size_update = gr.update(interactive=False, value=None)
    step_gamma_update = gr.update(interactive=False, value=None)
    polynomial_power_update = gr.update(interactive=False, value=None)

    if scheduler_type == "none":
        pass
    elif scheduler_type == "cosine":
        warmup_update = gr.update(interactive=True, value=_val("warmup_iters"))
        lr_decay_update = gr.update(interactive=True, value=_val("lr_decay_iters"))
        min_lr_update = gr.update(interactive=True, value=_val("min_lr"))
    elif scheduler_type == "constant_with_warmup":
        warmup_update = gr.update(interactive=True, value=_val("warmup_iters"))
    elif scheduler_type == "linear":
        warmup_update = gr.update(interactive=True, value=_val("warmup_iters"))
        lr_decay_update = gr.update(interactive=True, value=_val("lr_decay_iters"))
        min_lr_update = gr.update(interactive=True, value=_val("min_lr"))
    elif scheduler_type == "step":
        warmup_update = gr.update(interactive=True, value=_val("warmup_iters"))
        min_lr_update = gr.update(interactive=True, value=_val("min_lr"))
        step_size_update = gr.update(interactive=True, value=_val("step_size"))
        step_gamma_update = gr.update(interactive=True, value=_val("step_gamma"))
    elif scheduler_type == "polynomial":
        warmup_update = gr.update(interactive=True, value=_val("warmup_iters"))
        lr_decay_update = gr.update(interactive=True, value=_val("lr_decay_iters"))
        min_lr_update = gr.update(interactive=True, value=_val("min_lr"))
        polynomial_power_update = gr.update(interactive=True, value=_val("polynomial_power"))

    return [
        warmup_update,
        lr_decay_update,
        min_lr_update,
        step_size_update,
        step_gamma_update,
        polynomial_power_update,
    ]


def sft_reset_validation(lang_code):
    T_current = LANG_JSON[lang_code]
    return T_current["sft_no_dataset"], [], gr.update(interactive=False)


def sft_load_dataset(file_obj, dir_path, lang_code):
    T_current = LANG_JSON[lang_code]

    # Reset status
    msg = T_current["sft_no_dataset"]
    dataset = []

    # Prioritize file upload
    if file_obj is not None:
        dataset, msg = load_sft_dataset(file_path=file_obj.name)
    elif dir_path and dir_path.strip():
        dataset, msg = load_sft_dataset(dir_path=dir_path)

    is_valid, _ = validate_alpaca_format(dataset)
    status_val = T_current["sft_valid_format"] if is_valid else f"{T_current['sft_invalid_format']}: {msg}"

    return status_val, dataset, gr.update(interactive=is_valid)


def sft_train_cb(
    model_selection, dataset, epochs, lr, batch_size,
    max_seq_len, grad_acc,
    lr_scheduler_type, warmup_iters, lr_decay_iters, min_lr,
    step_size, step_gamma, polynomial_power,
    label_smoothing, freeze_layers, grad_clip, weight_decay,
    system_prompt,
    init_from,
    save_best_loss_ckpt,
    lang_code,
):
    T_current = LANG_JSON[lang_code]
    log_lines = []

    def _append_log_line(msg: str):
        if msg is None:
            return ""
        msg_s = str(msg)
        log_lines.append(msg_s)
        log_html = "<br>".join(log_lines)
        log_html += (
            "<script>"
            "const box=document.getElementById('sft-log-box');"
            "if(box){box.scrollTop=box.scrollHeight;}"
            "</script>"
        )
        return log_html

    # Pre-SFT training cleanup: Clear any residual GPU resources from previous failed training
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cache = ModelCache()
        cache.clear_cache()
        print("🧹 Pre-SFT training cleanup completed")
    except Exception as cleanup_err:
        print(f"Warning: Pre-SFT training cleanup encountered an issue: {cleanup_err}")

    empty_plot = generate_loss_chart_html([], [])

    if not model_selection or " - " not in model_selection:
        yield f"<div style='color:red;'>\u274c Please select a base model</div>", "", empty_plot, gr.update(interactive=True), gr.update(interactive=False)
        return

    model_id = int(model_selection.split(" - ")[0])
    model_info = dbm.get_model(model_id)
    if not model_info:
        yield f"<div style='color:red;'>\u274c Model not found</div>", "", empty_plot, gr.update(interactive=True), gr.update(interactive=False)
        return

    base_ckpt_path = os.path.join(model_info["out_dir"], "ckpt.pt")

    defaults_sft = DEFAULT_CONFIG["sft"]

    def safe_int(v, default_val):
        try:
            return default_val if v == "" or v is None else int(float(v))
        except (ValueError, TypeError):
            return default_val

    def safe_float(v, default_val):
        try:
            return default_val if v == "" or v is None else float(v)
        except (ValueError, TypeError):
            return default_val

    epochs = safe_int(epochs, defaults_sft["epochs"])
    batch_size = safe_int(batch_size, defaults_sft["batch_size"])
    max_seq_len = safe_int(max_seq_len, defaults_sft["max_seq_length"])
    grad_acc = safe_int(grad_acc, defaults_sft["gradient_accumulation_steps"])
    warmup_iters = safe_int(warmup_iters, defaults_sft["warmup_iters"])
    lr_decay_iters = safe_int(lr_decay_iters, defaults_sft["lr_decay_iters"])
    step_size = safe_int(step_size, defaults_sft["step_size"])
    freeze_layers = safe_int(freeze_layers, defaults_sft["freeze_layers"])
    lr = safe_float(lr, defaults_sft["learning_rate"])
    min_lr = safe_float(min_lr, defaults_sft["min_lr"])
    step_gamma = safe_float(step_gamma, defaults_sft["step_gamma"])
    polynomial_power = safe_float(polynomial_power, defaults_sft["polynomial_power"])
    label_smoothing = safe_float(label_smoothing, defaults_sft["label_smoothing"])
    grad_clip = safe_float(grad_clip, defaults_sft["grad_clip"])
    weight_decay = safe_float(weight_decay, defaults_sft["weight_decay"])
    lr_scheduler_type = lr_scheduler_type or defaults_sft["lr_scheduler_type"]
    init_from = init_from or defaults_sft["init_from"]
    save_best_loss_ckpt = bool(save_best_loss_ckpt)

    try:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_seq_len <= 0:
            raise ValueError("max_seq_length must be positive")
        if grad_acc <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if lr <= 0:
            raise ValueError("learning_rate must be positive")
        if min_lr < 0:
            raise ValueError("min_lr must be non-negative")
        if label_smoothing < 0 or label_smoothing >= 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if freeze_layers < 0:
            raise ValueError("freeze_layers must be non-negative")
        if grad_clip < 0:
            raise ValueError("grad_clip must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if lr_scheduler_type not in ["none", "cosine", "constant_with_warmup", "linear", "step", "polynomial"]:
            raise ValueError(f"Unsupported lr_scheduler_type: {lr_scheduler_type}")
        if init_from not in ["scratch", "resume"]:
            raise ValueError("init_from must be 'scratch' or 'resume'")
    except ValueError as e:
        yield f"<div style='color:red;'>\u274c {str(e)}</div>", "", empty_plot, gr.update(interactive=True), gr.update(interactive=False)
        return

    sft_cfg = {
        "epochs": epochs,
        "learning_rate": lr,
        "batch_size": batch_size,
        "max_seq_length": max_seq_len,
        "gradient_accumulation_steps": grad_acc,
        "lr_scheduler_type": lr_scheduler_type,
        "warmup_iters": warmup_iters,
        "lr_decay_iters": lr_decay_iters,
        "min_lr": min_lr,
        "step_size": step_size,
        "step_gamma": step_gamma,
        "polynomial_power": polynomial_power,
        "label_smoothing": label_smoothing,
        "freeze_layers": freeze_layers,
        "grad_clip": grad_clip,
        "weight_decay": weight_decay,
        "system_prompt": system_prompt,
        "init_from": init_from,
        "save_best_loss_checkpoint": save_best_loss_ckpt,
    }
    dbm.save_sft_config(model_id, sft_cfg)

    if not dataset:
        yield f"<div style='color:red;'>{T_current['sft_no_dataset']}</div>", "", empty_plot, gr.update(interactive=True), gr.update(interactive=False)
        return

    # Create SFT output directory
    sft_out_dir = os.path.join(model_info["out_dir"], "sft")
    os.makedirs(sft_out_dir, exist_ok=True)

    start_log_html = _append_log_line("🚀 Starting SFT Training...")
    yield make_progress_html(0, 100), start_log_html, empty_plot

    try:
        generator = sft_train_generator(
            base_model_ckpt_path=base_ckpt_path,
            data_dir=model_info["processed_data_dir"],
            dataset=dataset,
            out_dir=sft_out_dir,
            model_id=model_id,
            init_from=init_from,
            save_best_loss_checkpoint=save_best_loss_ckpt,
            epochs=int(epochs),
            learning_rate=lr,
            batch_size=int(batch_size),
            max_seq_length=int(max_seq_len),
            gradient_accumulation_steps=int(grad_acc),
            lr_scheduler_type=lr_scheduler_type,
            warmup_iters=warmup_iters,
            lr_decay_iters=lr_decay_iters,
            min_lr=min_lr,
            step_size=step_size,
            step_gamma=step_gamma,
            polynomial_power=polynomial_power,
            label_smoothing=label_smoothing,
            freeze_layers=freeze_layers,
            grad_clip=grad_clip,
            weight_decay=weight_decay,
            system_prompt=system_prompt,
        )

        for progress_html, log_msg, plot_data in generator:
            # Plot data format: (steps, losses, val_steps, val_losses)
            if plot_data and len(plot_data) >= 2:
                train_data = list(zip(plot_data[0], plot_data[1]))
                val_data = list(zip(plot_data[2], plot_data[3])) if len(plot_data) > 3 else []
                plot_html = generate_loss_chart_html(train_data, val_data)
            else:
                plot_html = empty_plot

            log_html = _append_log_line(log_msg)
            yield progress_html, log_html, plot_html, gr.update(interactive=False), gr.update(interactive=True)

        # Training completed, restore button states
        yield progress_html, log_html, plot_html, gr.update(interactive=True), gr.update(interactive=False)
    except Exception as e:
        import traceback

        print(f"SFT Training callback error: {traceback.format_exc()}")
        err_msg = f"Runtime Error in SFT Training: {str(e)}"

        # Critical: Clean up GPU resources after SFT training failure
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            try:
                cache = ModelCache()
                cache.clear_cache()
            except Exception as cache_err:
                print(f"Warning: Failed to clear model cache: {cache_err}")
            print("🧹 Post-SFT-error cleanup completed - GPU resources released")
        except Exception as cleanup_err:
            print(f"Warning: Post-SFT-error cleanup failed: {cleanup_err}")

        err_log_html = _append_log_line(f"<div style='color:red;'>{err_msg}</div>")
        yield f"<div style='color:red;'>{err_msg}</div>", err_log_html, empty_plot, gr.update(interactive=True), gr.update(interactive=False)


def sft_stop_cb():
    stop_sft_training()
    # Return: log message, sft_start_btn (enabled), sft_stop_btn (disabled)
    return "🛑 Stopping SFT...", gr.update(interactive=True), gr.update(interactive=False)
