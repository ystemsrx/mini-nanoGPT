from src.config import DEFAULT_CONFIG
from src.ui.callbacks.training import (
    training_cb,
    stop_training_cb,
    update_lr_scheduler_params,
    update_self_attention_params,
)


def bind_training(
    demo,
    stop_btn,
    lr_scheduler_box,
    warmup_box,
    lr_decay_box,
    min_lr_box,
    step_size_box,
    step_gamma_box,
    polynomial_power_box,
    use_self_attention_box,
    ffn_hidden_mult_box,
    qkv_bias_box,
    attn_dropout_box,
    resid_dropout_box,
    ln_eps_box,
    init_std_box,
    use_flash_attn_box,
    pos_encoding_type_box,
    rope_base_box,
    rope_cache_size_box,
    alibi_bias_scale_box,
    ffn_activation_box,
    attention_scale_factor_box,
    gradient_checkpointing_box,
    cache_strategy_box,
    max_cache_size_box,
    strict_validation_box,
    fallback_on_error_box,
    train_btn,
    data_dir_box,
    out_dir_box,
    plot_interval_box,
    log_interval_box,
    num_eval_seeds_box,
    save_best_val_ckpt_box,
    init_from_box,
    grad_acc_box,
    batch_size_box,
    block_size_box,
    n_layer_box,
    n_head_box,
    n_embd_box,
    dropout_box,
    bias_box,
    lr_box,
    max_iters_box,
    weight_decay_box,
    beta1_box,
    beta2_box,
    backend_box,
    device_box,
    dtype_box,
    compile_box,
    seed_box,
    save_interval_box,
    train_progress,
    train_log,
    train_plot,
):
    stop_btn.click(fn=stop_training_cb, inputs=[], outputs=[train_log, train_btn, stop_btn])

    # -----------------------------
    # LR Scheduler Callback
    # -----------------------------
    lr_scheduler_box.change(
        fn=update_lr_scheduler_params,
        inputs=[lr_scheduler_box],
        outputs=[
            warmup_box,
            lr_decay_box,
            min_lr_box,
            step_size_box,
            step_gamma_box,
            polynomial_power_box,
        ],
    )

    # -----------------------------
    # Self-Attention Parameters Callback
    # -----------------------------
    use_self_attention_box.change(
        fn=update_self_attention_params,
        inputs=[use_self_attention_box],
        outputs=[
            ffn_hidden_mult_box,
            qkv_bias_box,
            attn_dropout_box,
            resid_dropout_box,
            ln_eps_box,
            init_std_box,
            use_flash_attn_box,
            pos_encoding_type_box,
            rope_base_box,
            # New optimized parameters
            rope_cache_size_box,
            alibi_bias_scale_box,
            ffn_activation_box,
            attention_scale_factor_box,
            gradient_checkpointing_box,
            cache_strategy_box,
            max_cache_size_box,
            strict_validation_box,
            fallback_on_error_box,
        ],
    )

    # ------------------------------------------------------------------ #
    # Call backs: start training
    # ------------------------------------------------------------------ #
    train_btn.click(
        fn=training_cb,
        inputs=[
            data_dir_box,
            out_dir_box,
            plot_interval_box,
            log_interval_box,
            num_eval_seeds_box,
            save_best_val_ckpt_box,
            init_from_box,
            grad_acc_box,
            batch_size_box,
            block_size_box,
            n_layer_box,
            n_head_box,
            n_embd_box,
            dropout_box,
            bias_box,
            lr_box,
            max_iters_box,
            weight_decay_box,
            beta1_box,
            beta2_box,
            lr_scheduler_box,
            warmup_box,
            lr_decay_box,
            min_lr_box,
            step_size_box,
            step_gamma_box,
            polynomial_power_box,
            backend_box,
            device_box,
            dtype_box,
            compile_box,
            seed_box,
            save_interval_box,
            # Self-attention parameters
            use_self_attention_box,
            ffn_hidden_mult_box,
            qkv_bias_box,
            attn_dropout_box,
            resid_dropout_box,
            ln_eps_box,
            init_std_box,
            use_flash_attn_box,
            pos_encoding_type_box,
            rope_base_box,
            # New optimized parameters
            rope_cache_size_box,
            alibi_bias_scale_box,
            ffn_activation_box,
            attention_scale_factor_box,
            gradient_checkpointing_box,
            cache_strategy_box,
            max_cache_size_box,
            strict_validation_box,
            fallback_on_error_box,
        ],
        outputs=[train_progress, train_log, train_plot, train_btn, stop_btn],
    )

    # Initialize LR scheduler params display logic on app load
    demo.load(
        fn=lambda: update_lr_scheduler_params(DEFAULT_CONFIG["training"]["lr_scheduler_type"]),
        inputs=None,
        outputs=[
            warmup_box,
            lr_decay_box,
            min_lr_box,
            step_size_box,
            step_gamma_box,
            polynomial_power_box,
        ],
        queue=False,
    )

    # Manually update LR scheduler params when changed
    lr_scheduler_box.change(
        fn=update_lr_scheduler_params,
        inputs=[lr_scheduler_box],
        outputs=[
            warmup_box,
            lr_decay_box,
            min_lr_box,
            step_size_box,
            step_gamma_box,
            polynomial_power_box,
        ],
        queue=False,
    )
