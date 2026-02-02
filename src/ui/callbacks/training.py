import torch
import gradio as gr

from src.config import DEFAULT_CONFIG
from src.train import train_model_generator, stop_training
from src.infer_cache import ModelCache
from src.ui.charts import generate_loss_chart_html


def stop_training_cb():
    """Stop training and return button states"""
    stop_training()
    # Return: log message, train_btn (enabled), stop_btn (disabled)
    return "🛑 Stopping training...", gr.update(interactive=True), gr.update(interactive=False)


def update_lr_scheduler_params(scheduler_type):
    defaults_train = DEFAULT_CONFIG["training"]

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
        warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
        lr_decay_update = gr.update(interactive=True, value=defaults_train["lr_decay_iters"])
        min_lr_update = gr.update(interactive=True, value=defaults_train["min_lr"])
    elif scheduler_type == "constant_with_warmup":
        warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
    elif scheduler_type == "linear":
        warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
        lr_decay_update = gr.update(interactive=True, value=defaults_train["lr_decay_iters"])
        min_lr_update = gr.update(interactive=True, value=defaults_train["min_lr"])
    elif scheduler_type == "step":
        warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
        min_lr_update = gr.update(interactive=True, value=defaults_train["min_lr"])
        step_size_update = gr.update(interactive=True, value=defaults_train["step_size"])
        step_gamma_update = gr.update(interactive=True, value=defaults_train["step_gamma"])
    elif scheduler_type == "polynomial":
        warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
        lr_decay_update = gr.update(interactive=True, value=defaults_train["lr_decay_iters"])
        min_lr_update = gr.update(interactive=True, value=defaults_train["min_lr"])
        polynomial_power_update = gr.update(interactive=True, value=defaults_train["polynomial_power"])

    return [
        warmup_update, lr_decay_update, min_lr_update,
        step_size_update, step_gamma_update, polynomial_power_update,
    ]


def update_self_attention_params(use_self_attention):
    """Update visibility of self-attention parameters based on checkbox"""
    defaults_train = DEFAULT_CONFIG["training"]

    if use_self_attention:
        return [
            gr.update(visible=True, value=defaults_train["ffn_hidden_mult"]),
            gr.update(visible=True, value=defaults_train["qkv_bias"]),
            gr.update(visible=True, value=defaults_train["attn_dropout"]),
            gr.update(visible=True, value=defaults_train["resid_dropout"]),
            gr.update(visible=True, value=defaults_train["ln_eps"]),
            gr.update(visible=True, value=defaults_train["init_std"]),
            gr.update(visible=True, value=defaults_train["use_flash_attn"]),
            gr.update(visible=True, value=defaults_train["pos_encoding_type"]),
            gr.update(visible=True, value=defaults_train["rope_base"]),
            # New optimized parameters
            gr.update(visible=True, value=defaults_train["rope_cache_size"]),
            gr.update(visible=True, value=defaults_train["alibi_bias_scale"]),
            gr.update(visible=True, value=defaults_train["ffn_activation"]),
            gr.update(visible=True, value=defaults_train["attention_scale_factor"]),
            gr.update(visible=True, value=defaults_train["gradient_checkpointing"]),
            gr.update(visible=True, value=defaults_train["cache_strategy"]),
            gr.update(visible=True, value=defaults_train["max_cache_size"]),
            gr.update(visible=True, value=defaults_train["strict_validation"]),
            gr.update(visible=True, value=defaults_train["fallback_on_error"]),
        ]

    return [
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        # New optimized parameters
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=False),
    ]


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
    seed_, save_interval_,
    # Self-attention parameters
    use_self_attention_, ffn_hidden_mult_, qkv_bias_, attn_dropout_,
    resid_dropout_, ln_eps_, init_std_, use_flash_attn_, pos_encoding_type_, rope_base_,
    # New optimized parameters
    rope_cache_size_, alibi_bias_scale_, ffn_activation_, attention_scale_factor_,
    gradient_checkpointing_, cache_strategy_, max_cache_size_, strict_validation_, fallback_on_error_,
):
    # Pre-training cleanup: Clear any residual GPU resources from previous failed training
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # Also clear model cache to ensure fresh start
        cache = ModelCache()
        cache.clear_cache()
        print("🧹 Pre-training cleanup completed")
    except Exception as cleanup_err:
        print(f"Warning: Pre-training cleanup encountered an issue: {cleanup_err}")

    empty_plot_html = generate_loss_chart_html([], [])

    # Enhanced input validation with better error messages
    try:
        # Convert Number inputs safely with improved error handling
        num_eval_seeds_int = int(float(num_eval_seeds_))
        if not (0 <= num_eval_seeds_int <= 2**32 - 1):
            raise ValueError(f"num_eval_seeds must be between 0 and {2**32 - 1}, got {num_eval_seeds_int}")

        # Validate new parameters
        rope_cache_size_int = None if rope_cache_size_ == 0 else int(float(rope_cache_size_))
        if rope_cache_size_int is not None and rope_cache_size_int < 0:
            raise ValueError(f"rope_cache_size must be non-negative or 0 for auto, got {rope_cache_size_int}")

        alibi_bias_scale_float = float(alibi_bias_scale_)
        if alibi_bias_scale_float <= 0:
            raise ValueError(f"alibi_bias_scale must be positive, got {alibi_bias_scale_float}")

        attention_scale_factor_float = float(attention_scale_factor_)
        if attention_scale_factor_float <= 0:
            raise ValueError(f"attention_scale_factor must be positive, got {attention_scale_factor_float}")

        max_cache_size_int = int(float(max_cache_size_))
        if max_cache_size_int <= 0:
            raise ValueError(f"max_cache_size must be positive, got {max_cache_size_int}")

        # Validate FFN activation
        if ffn_activation_ not in ["gelu", "relu", "swish"]:
            raise ValueError(f"ffn_activation must be one of ['gelu', 'relu', 'swish'], got {ffn_activation_}")

        # Validate cache strategy
        if cache_strategy_ not in ["adaptive", "fixed", "minimal"]:
            raise ValueError(f"cache_strategy must be one of ['adaptive', 'fixed', 'minimal'], got {cache_strategy_}")

    except ValueError as e:
        error_msg = f"Parameter validation error: {str(e)}"
        error_html = f"<span style='color:red;'>{error_msg}</span>"
        yield (f"<div style='color:red;'>{error_msg}</div>", error_html, empty_plot_html, gr.update(interactive=True), gr.update(interactive=False))
        return

    try:
        defaults_train = DEFAULT_CONFIG["training"]

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

        def safe_bool(v, default_val):
            try:
                return default_val if v is None else bool(v)
            except (ValueError, TypeError):
                return default_val

        def safe_str(v, default_val):
            try:
                return default_val if v is None or v == "" else str(v)
            except (ValueError, TypeError):
                return default_val

        gen = train_model_generator(
            data_dir=data_dir_,
            out_dir=out_dir_,
            plot_interval=safe_int(plot_interval_, defaults_train["plot_interval"]),
            log_interval=safe_int(log_interval_, defaults_train["log_interval"]),
            num_eval_seeds=num_eval_seeds_int,
            save_best_val_checkpoint=bool(save_best_val_ckpt_),
            init_from=init_from_,
            gradient_accumulation_steps=safe_int(grad_acc_, defaults_train["gradient_accumulation_steps"]),
            batch_size=safe_int(batch_size_, defaults_train["batch_size"]),
            block_size=safe_int(block_size_, defaults_train["block_size"]),
            n_layer=safe_int(n_layer_, defaults_train["n_layer"]),
            n_head=safe_int(n_head_, defaults_train["n_head"]),
            n_embd=safe_int(n_embd_, defaults_train["n_embd"]),
            dropout=safe_float(dropout_, defaults_train["dropout"]),
            bias=bool(bias_),
            learning_rate=safe_float(lr_, defaults_train["learning_rate"]),
            max_iters=safe_int(max_iters_, defaults_train["max_iters"]),
            weight_decay=safe_float(weight_decay_, defaults_train["weight_decay"]),
            beta1=safe_float(beta1_, defaults_train["beta1"]),
            beta2=safe_float(beta2_, defaults_train["beta2"]),
            lr_scheduler_type=lr_scheduler_type_,
            warmup_iters=safe_int(warmup_, defaults_train["warmup_iters"]),
            lr_decay_iters=safe_int(lr_decay_, defaults_train["lr_decay_iters"]),
            min_lr=safe_float(min_lr_, defaults_train["min_lr"]),
            step_size=safe_int(step_size_, defaults_train["step_size"]),
            step_gamma=safe_float(step_gamma_, defaults_train["step_gamma"]),
            polynomial_power=safe_float(polynomial_power_, defaults_train["polynomial_power"]),
            backend=safe_str(backend_, defaults_train["backend"]),
            device=safe_str(device_, defaults_train["device"]),
            dtype=safe_str(dtype_, defaults_train["dtype"]),
            compile_model=bool(compile_),
            seed=safe_int(seed_, defaults_train["seed"]),
            save_interval=safe_int(save_interval_, defaults_train["save_interval"]),
            # Self-attention parameters
            use_self_attention=safe_bool(use_self_attention_, defaults_train["use_self_attention"]),
            ffn_hidden_mult=safe_int(ffn_hidden_mult_, defaults_train["ffn_hidden_mult"]),
            qkv_bias=safe_bool(qkv_bias_, defaults_train["qkv_bias"]),
            attn_dropout=safe_float(attn_dropout_, defaults_train["attn_dropout"]),
            resid_dropout=safe_float(resid_dropout_, defaults_train["resid_dropout"]),
            ln_eps=safe_float(ln_eps_, defaults_train["ln_eps"]),
            init_std=safe_float(init_std_, defaults_train["init_std"]),
            use_flash_attn=safe_bool(use_flash_attn_, defaults_train["use_flash_attn"]),
            pos_encoding_type=pos_encoding_type_ if pos_encoding_type_ else defaults_train["pos_encoding_type"],
            rope_base=safe_int(rope_base_, defaults_train["rope_base"]),
            # New optimized parameters
            rope_cache_size=rope_cache_size_int,
            alibi_bias_scale=alibi_bias_scale_float,
            ffn_activation=safe_str(ffn_activation_, defaults_train["ffn_activation"]),
            attention_scale_factor=attention_scale_factor_float,
            gradient_checkpointing=safe_bool(gradient_checkpointing_, defaults_train["gradient_checkpointing"]),
            cache_strategy=safe_str(cache_strategy_, defaults_train["cache_strategy"]),
            max_cache_size=max_cache_size_int,
            strict_validation=safe_bool(strict_validation_, defaults_train["strict_validation"]),
            fallback_on_error=safe_bool(fallback_on_error_, defaults_train["fallback_on_error"]),
        )

        for p_html_progress, log_line_html, plot_data_tuple in gen:
            current_plot_rendered_html = empty_plot_html
            if plot_data_tuple:
                tr_steps, tr_losses, val_steps, val_losses = plot_data_tuple
                train_data_tuples = list(zip(tr_steps, tr_losses)) if tr_steps and tr_losses else []
                val_data_tuples = list(zip(val_steps, val_losses)) if val_steps and val_losses else []
                current_plot_rendered_html = generate_loss_chart_html(train_data_tuples, val_data_tuples)

            # Wrap error messages in red color
            if log_line_html and (log_line_html.startswith("Error:") or log_line_html.startswith("❌")):
                log_line_html = f"<span style='color:red;'>{log_line_html}</span>"

            yield (p_html_progress, log_line_html, current_plot_rendered_html, gr.update(interactive=False), gr.update(interactive=True))

        # Training completed, restore button states
        yield (p_html_progress, log_line_html, current_plot_rendered_html, gr.update(interactive=True), gr.update(interactive=False))

    except Exception as e:
        import traceback

        print(f"Training callback error: {traceback.format_exc()}")
        err_msg = f"Runtime Error in Training: {str(e)}"

        # Critical: Clean up GPU resources after training failure
        try:
            import gc

            # Force garbage collection to release Python objects
            gc.collect()

            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Clear model cache to ensure next training starts fresh
            try:
                cache = ModelCache()
                cache.clear_cache()
            except Exception as cache_err:
                print(f"Warning: Failed to clear model cache: {cache_err}")

            print("🧹 Post-error cleanup completed - GPU resources released")
        except Exception as cleanup_err:
            print(f"Warning: Post-error cleanup failed: {cleanup_err}")

        error_html = f"<span style='color:red;'>{err_msg}</span>"
        yield (f"<div style='color:red;'>{err_msg}</div>", error_html, empty_plot_html, gr.update(interactive=True), gr.update(interactive=False))
