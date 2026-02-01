import os
import pickle

import gradio as gr

from src.config import DEFAULT_CONFIG
from src.ui.charts import generate_loss_chart_html
from src.ui.helpers import _create_plot_html_from_log, _get_model_choices_list
from src.ui.state import dbm
from src.ui.callbacks.training import update_lr_scheduler_params, update_self_attention_params


def _reset_updates():
    def _d(val=""):
        return gr.update(value=val)

    def _b(val=False):
        return gr.update(value=val)

    d_train = DEFAULT_CONFIG["training"]
    d_inf = DEFAULT_CONFIG["inference"]

    # Get default scheduler state
    default_scheduler = d_train["lr_scheduler_type"]
    warmup_update, lr_decay_update, min_lr_update, step_size_update, step_gamma_update, polynomial_power_update = (
        update_lr_scheduler_params(default_scheduler)
    )

    # Get default self-attn state
    default_use_self_attention = d_train["use_self_attention"]
    self_attn_updates = update_self_attention_params(default_use_self_attention)

    base_updates = [
        _b(True),
        _d("new_model"),
        _d(),
        _d(),
        _d(d_train["plot_interval"]),
        _d(d_train["log_interval"]),
        _d(d_train["num_eval_seeds"]),
        _b(d_train["save_best_val_checkpoint"]),
        _d(d_train["init_from"]),
        _d(d_train["gradient_accumulation_steps"]),
        _d(d_train["batch_size"]),
        _d(d_train["block_size"]),
        _d(d_train["n_layer"]),
        _d(d_train["n_head"]),
        _d(d_train["n_embd"]),
        _d(d_train["dropout"]),
        _b(d_train["bias"]),
        _d(d_train["learning_rate"]),
        _d(d_train["max_iters"]),
        _d(d_train["weight_decay"]),
        _d(d_train["beta1"]),
        _d(d_train["beta2"]),
        _d(d_train["lr_scheduler_type"]),
        warmup_update,
        lr_decay_update,
        min_lr_update,
        step_size_update,
        step_gamma_update,
        polynomial_power_update,
        _d(d_train["backend"]),
        _d(d_train["device"]),
        _d(d_train["dtype"]),
        _b(d_train["compile_model"]),
        _d(d_train["seed"]),
        _d(d_train["save_interval"]),
        # Self-attention parameters
        _b(d_train["use_self_attention"]),
        self_attn_updates[0],
        self_attn_updates[1],
        self_attn_updates[2],
        self_attn_updates[3],
        self_attn_updates[4],
        self_attn_updates[5],
        self_attn_updates[6],
        self_attn_updates[7],
        self_attn_updates[8],
        # New optimized parameters
        self_attn_updates[9],
        self_attn_updates[10],
        self_attn_updates[11],
        self_attn_updates[12],
        self_attn_updates[13],
        self_attn_updates[14],
        self_attn_updates[15],
        self_attn_updates[16],
        self_attn_updates[17],
        generate_loss_chart_html([], []),
        "",
        _d(),
        _d(),
        _d(d_inf["prompt"]),
        _d(d_inf["num_samples"]),
        _d(d_inf["max_new_tokens"]),
        _d(d_inf["temperature"]),
        _d(d_inf["top_k"]),
        _d(d_inf["dtype"]),
        _d(d_inf["device"]),
        _d(d_inf["seed"]),
        gr.update(),
        "",
        "",
        gr.update(value=[]),
        gr.update(value=DEFAULT_CONFIG["sft"]["system_prompt"]),
        "",
    ]

    comparison_updates = [
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        _d(d_inf["num_samples"]),
        _d(d_inf["max_new_tokens"]),
        _d(d_inf["temperature"]),
        _d(d_inf["top_k"]),
        _d(d_inf["dtype"]),
        _d(d_inf["seed"]),
        _d(d_inf["num_samples"]),
        _d(d_inf["max_new_tokens"]),
        _d(d_inf["temperature"]),
        _d(d_inf["top_k"]),
        _d(d_inf["dtype"]),
        _d(d_inf["seed"]),
        _d(d_inf["prompt"]),
        gr.update(),
        gr.update(value=""),
        gr.update(value=""),
        _d(),
        _d(),
        _d(),
        _d(),
    ]

    return base_updates + comparison_updates


def select_model_cb(sel: str):
    if not sel:
        return _reset_updates()
    try:
        mid = int(sel.split(" - ")[0])
    except ValueError:
        return _reset_updates()

    cfg = dbm.get_training_config(mid) or {}
    icfg = dbm.get_inference_config(mid) or {}
    info = dbm.get_model_basic_info(mid) or {}
    name = info.get("name", "unknown_model")

    # Use relative paths for portability
    if "dir_path" in info:
        out_dir_root = info["dir_path"]
        folder = os.path.basename(out_dir_root)
        data_processed_dir = os.path.join("data", folder, "processed")
    else:
        folder_name_part = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in name)
        folder = f"{folder_name_part}_{mid}"
        data_processed_dir = os.path.join("data", folder, "processed")
        out_dir_root = os.path.join("out", folder)

    def _cfg(k, default_val_from_const):
        return cfg.get(k, default_val_from_const)

    def _ic(k, default_val_from_const):
        return icfg.get(k, default_val_from_const)

    loss_log_path = dbm.get_training_log_path(mid)
    loss_plot_html_content = _create_plot_html_from_log(loss_log_path)

    train_log_s = ""
    if loss_log_path and os.path.exists(loss_log_path):
        try:
            with open(loss_log_path, "rb") as f:
                log_data_dict = pickle.load(f)

            log_tr_steps = log_data_dict.get("train_plot_steps", [])
            log_tr_losses = log_data_dict.get("train_plot_losses", [])
            log_val_steps = log_data_dict.get("val_plot_steps", [])
            log_val_losses = log_data_dict.get("val_plot_losses", [])

            log_lines = []
            if log_tr_steps:
                log_lines.append(f"Training history for model {mid} - {name}:")
                val_map = dict(zip(log_val_steps, log_val_losses)) if log_val_steps and log_val_losses else {}
                for i, (step, loss) in enumerate(zip(log_tr_steps, log_tr_losses)):
                    line = f"Step {step}: train_loss={loss:.4f}"
                    if step in val_map:
                        line += f", val_loss={val_map[step]:.4f}"
                    log_lines.append(line)
                    if i >= 199:
                        log_lines.append(f"... (showing first 200 of {len(log_tr_steps)} records)")
                        break
            train_log_s = "\n".join(log_lines)
        except Exception as e_log:
            train_log_s = f"Error loading training log details: {str(e_log)}"

    d_train_defaults = DEFAULT_CONFIG["training"]
    d_inf_defaults = DEFAULT_CONFIG["inference"]

    # Get full inference history including HTML formatted output
    inference_history_data = dbm.get_inference_history_full(mid)
    if inference_history_data.get("html_content"):
        inference_history_html = inference_history_data["html_content"]
    elif inference_history_data.get("content"):
        plain_text = inference_history_data["content"]
        escaped_text = (
            plain_text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
        )
        inference_history_html = (
            "<div style=\"font-family: monospace; white-space: pre-wrap; line-height: 1.6; padding: 10px; "
            "background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0;\">"
            f"{escaped_text}</div>"
        )
    else:
        inference_history_html = ""

    # Get chat history
    chat_history_data = dbm.get_chat_history(mid)
    chat_history = chat_history_data.get("history", [])
    chat_advanced_html_saved = chat_history_data.get("advanced_html", "")
    chat_system_prompt_saved = chat_history_data.get("system_prompt", "")

    inference_advanced_html = inference_history_data.get("advanced_html", "")

    scheduler_type = _cfg("lr_scheduler_type", d_train_defaults["lr_scheduler_type"])
    warmup_update, lr_decay_update, min_lr_update, step_size_update, step_gamma_update, polynomial_power_update = (
        update_lr_scheduler_params(scheduler_type)
    )

    use_self_attention = _cfg("use_self_attention", d_train_defaults["use_self_attention"])
    self_attn_updates = update_self_attention_params(use_self_attention)

    base_updates = [
        gr.update(value=False),
        gr.update(value=name),
        gr.update(value=data_processed_dir),
        gr.update(value=out_dir_root),
        gr.update(value=_cfg("plot_interval", d_train_defaults["plot_interval"])),
        gr.update(value=_cfg("log_interval", d_train_defaults["log_interval"])),
        gr.update(value=_cfg("num_eval_seeds", d_train_defaults["num_eval_seeds"])),
        gr.update(value=bool(_cfg("save_best_val_checkpoint", d_train_defaults["save_best_val_checkpoint"]))),
        gr.update(value=_cfg("init_from", d_train_defaults["init_from"])),
        gr.update(value=_cfg("gradient_accumulation_steps", d_train_defaults["gradient_accumulation_steps"])),
        gr.update(value=_cfg("batch_size", d_train_defaults["batch_size"])),
        gr.update(value=_cfg("block_size", d_train_defaults["block_size"])),
        gr.update(value=_cfg("n_layer", d_train_defaults["n_layer"])),
        gr.update(value=_cfg("n_head", d_train_defaults["n_head"])),
        gr.update(value=_cfg("n_embd", d_train_defaults["n_embd"])),
        gr.update(value=_cfg("dropout", d_train_defaults["dropout"])),
        gr.update(value=bool(_cfg("bias", d_train_defaults["bias"]))),
        gr.update(value=_cfg("learning_rate", d_train_defaults["learning_rate"])),
        gr.update(value=_cfg("max_iters", d_train_defaults["max_iters"])),
        gr.update(value=_cfg("weight_decay", d_train_defaults["weight_decay"])),
        gr.update(value=_cfg("beta1", d_train_defaults["beta1"])),
        gr.update(value=_cfg("beta2", d_train_defaults["beta2"])),
        gr.update(value=scheduler_type),
        warmup_update,
        lr_decay_update,
        min_lr_update,
        step_size_update,
        step_gamma_update,
        polynomial_power_update,
        gr.update(value=_cfg("backend", d_train_defaults["backend"])),
        gr.update(value=_cfg("device", d_train_defaults["device"])),
        gr.update(value=_cfg("dtype", d_train_defaults["dtype"])),
        gr.update(value=bool(_cfg("compile_model", d_train_defaults["compile_model"]))),
        gr.update(value=_cfg("seed", d_train_defaults["seed"])),
        gr.update(value=_cfg("save_interval", d_train_defaults["save_interval"])),
        # Self-attention parameters
        gr.update(value=use_self_attention),
        self_attn_updates[0] if use_self_attention else gr.update(visible=False, value=_cfg("ffn_hidden_mult", d_train_defaults["ffn_hidden_mult"])),
        self_attn_updates[1] if use_self_attention else gr.update(visible=False, value=_cfg("qkv_bias", d_train_defaults["qkv_bias"])),
        self_attn_updates[2] if use_self_attention else gr.update(visible=False, value=_cfg("attn_dropout", d_train_defaults["attn_dropout"])),
        self_attn_updates[3] if use_self_attention else gr.update(visible=False, value=_cfg("resid_dropout", d_train_defaults["resid_dropout"])),
        self_attn_updates[4] if use_self_attention else gr.update(visible=False, value=_cfg("ln_eps", d_train_defaults["ln_eps"])),
        self_attn_updates[5] if use_self_attention else gr.update(visible=False, value=_cfg("init_std", d_train_defaults["init_std"])),
        self_attn_updates[6] if use_self_attention else gr.update(visible=False, value=_cfg("use_flash_attn", d_train_defaults["use_flash_attn"])),
        self_attn_updates[7] if use_self_attention else gr.update(visible=False, value=_cfg("pos_encoding_type", d_train_defaults["pos_encoding_type"])),
        self_attn_updates[8] if use_self_attention else gr.update(visible=False, value=_cfg("rope_base", d_train_defaults["rope_base"])),
        # New optimized parameters
        self_attn_updates[9] if use_self_attention else gr.update(visible=False, value=_cfg("rope_cache_size", d_train_defaults["rope_cache_size"])),
        self_attn_updates[10] if use_self_attention else gr.update(visible=False, value=_cfg("alibi_bias_scale", d_train_defaults["alibi_bias_scale"])),
        self_attn_updates[11] if use_self_attention else gr.update(visible=False, value=_cfg("ffn_activation", d_train_defaults["ffn_activation"])),
        self_attn_updates[12] if use_self_attention else gr.update(visible=False, value=_cfg("attention_scale_factor", d_train_defaults["attention_scale_factor"])),
        self_attn_updates[13] if use_self_attention else gr.update(visible=False, value=_cfg("gradient_checkpointing", d_train_defaults["gradient_checkpointing"])),
        self_attn_updates[14] if use_self_attention else gr.update(visible=False, value=_cfg("cache_strategy", d_train_defaults["cache_strategy"])),
        self_attn_updates[15] if use_self_attention else gr.update(visible=False, value=_cfg("max_cache_size", d_train_defaults["max_cache_size"])),
        self_attn_updates[16] if use_self_attention else gr.update(visible=False, value=_cfg("strict_validation", d_train_defaults["strict_validation"])),
        self_attn_updates[17] if use_self_attention else gr.update(visible=False, value=_cfg("fallback_on_error", d_train_defaults["fallback_on_error"])),
        loss_plot_html_content,
        train_log_s,
        gr.update(value=data_processed_dir),
        gr.update(value=out_dir_root),
        gr.update(value=_ic("prompt", d_inf_defaults["prompt"])),
        gr.update(value=_ic("num_samples", d_inf_defaults["num_samples"])),
        gr.update(value=_ic("max_new_tokens", d_inf_defaults["max_new_tokens"])),
        gr.update(value=_ic("temperature", d_inf_defaults["temperature"])),
        gr.update(value=_ic("top_k", d_inf_defaults["top_k"])),
        gr.update(value=_ic("dtype", d_inf_defaults["dtype"])),
        gr.update(value=_ic("device", d_inf_defaults["device"])),
        gr.update(value=_ic("seed", d_inf_defaults["seed"])),
        gr.update(),
        inference_history_html,
        inference_advanced_html,
        gr.update(value=chat_history),
        gr.update(value=chat_system_prompt_saved if chat_system_prompt_saved is not None else ""),
        chat_advanced_html_saved,
    ]

    comparison_updates = [
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(value=_ic("num_samples", d_inf_defaults["num_samples"])),
        gr.update(value=_ic("max_new_tokens", d_inf_defaults["max_new_tokens"])),
        gr.update(value=_ic("temperature", d_inf_defaults["temperature"])),
        gr.update(value=_ic("top_k", d_inf_defaults["top_k"])),
        gr.update(value=_ic("dtype", d_inf_defaults["dtype"])),
        gr.update(value=_ic("seed", d_inf_defaults["seed"])),
        gr.update(value=_ic("num_samples", d_inf_defaults["num_samples"])),
        gr.update(value=_ic("max_new_tokens", d_inf_defaults["max_new_tokens"])),
        gr.update(value=_ic("temperature", d_inf_defaults["temperature"])),
        gr.update(value=_ic("top_k", d_inf_defaults["top_k"])),
        gr.update(value=_ic("dtype", d_inf_defaults["dtype"])),
        gr.update(value=_ic("seed", d_inf_defaults["seed"])),
        gr.update(value=_ic("prompt", d_inf_defaults["prompt"])),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
    ]

    return base_updates + comparison_updates


def delete_model_cb(sel: str):
    if sel and " - " in sel:
        try:
            model_id = int(sel.split(" - ")[0])
            dbm.delete_model(model_id)
        except Exception as e:
            print(f"Error deleting model: {e}")

    updated_choices = _get_model_choices_list()

    main_dropdown_update = gr.update(choices=updated_choices, value=None)
    comp_left_update = gr.update(choices=updated_choices, value=None)
    comp_right_update = gr.update(choices=updated_choices, value=None)

    reset_values = _reset_updates()

    return [main_dropdown_update, comp_left_update, comp_right_update] + reset_values
