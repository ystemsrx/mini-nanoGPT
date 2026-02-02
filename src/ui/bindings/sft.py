from src.ui.callbacks.sft import update_sft_lr_scheduler_params, sft_reset_validation, sft_load_dataset, sft_train_cb, sft_stop_cb


def bind_sft(
    sft_validate_btn,
    sft_dataset_file,
    sft_dataset_dir,
    lang_select,
    sft_format_status,
    sft_dataset_state,
    sft_start_btn,
    model_dropdown,
    sft_epochs,
    sft_learning_rate,
    sft_batch_size,
    sft_max_seq_length,
    sft_gradient_accumulation,
    sft_lr_scheduler,
    sft_warmup_iters,
    sft_lr_decay_iters,
    sft_min_lr,
    sft_step_size,
    sft_step_gamma,
    sft_polynomial_power,
    sft_label_smoothing,
    sft_freeze_layers,
    sft_grad_clip,
    sft_weight_decay,
    sft_system_prompt,
    sft_init_from,
    sft_save_best_loss_ckpt,
    sft_progress,
    sft_log,
    sft_plot,
    sft_stop_btn,
    inf_chat_mode,
):
    sft_lr_scheduler.change(
        fn=update_sft_lr_scheduler_params,
        inputs=[sft_lr_scheduler],
        outputs=[
            sft_warmup_iters,
            sft_lr_decay_iters,
            sft_min_lr,
            sft_step_size,
            sft_step_gamma,
            sft_polynomial_power,
        ],
    )

    sft_validate_btn.click(
        fn=sft_load_dataset,
        inputs=[sft_dataset_file, sft_dataset_dir, lang_select],
        outputs=[sft_format_status, sft_dataset_state, sft_start_btn],
    )

    sft_dataset_file.change(
        fn=sft_reset_validation,
        inputs=[lang_select],
        outputs=[sft_format_status, sft_dataset_state, sft_start_btn],
    )

    sft_dataset_dir.change(
        fn=sft_reset_validation,
        inputs=[lang_select],
        outputs=[sft_format_status, sft_dataset_state, sft_start_btn],
    )

    sft_start_btn.click(
        fn=sft_train_cb,
        inputs=[
            model_dropdown,
            sft_dataset_state,
            sft_epochs,
            sft_learning_rate,
            sft_batch_size,
            sft_max_seq_length,
            sft_gradient_accumulation,
            sft_lr_scheduler,
            sft_warmup_iters,
            sft_lr_decay_iters,
            sft_min_lr,
            sft_step_size,
            sft_step_gamma,
            sft_polynomial_power,
            sft_label_smoothing,
            sft_freeze_layers,
            sft_grad_clip,
            sft_weight_decay,
            sft_system_prompt,
            sft_init_from,
            sft_save_best_loss_ckpt,
            lang_select,
        ],
        outputs=[sft_progress, sft_log, sft_plot, sft_start_btn, sft_stop_btn, inf_chat_mode],
    )

    sft_stop_btn.click(
        fn=sft_stop_cb,
        inputs=[],
        outputs=[sft_log, sft_start_btn, sft_stop_btn],
    )
