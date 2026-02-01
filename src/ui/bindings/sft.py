from src.ui.callbacks.sft import sft_refresh_models, sft_load_dataset, sft_train_cb, sft_stop_cb


def bind_sft(
    sft_refresh_model_btn,
    sft_base_model,
    sft_validate_btn,
    sft_dataset_file,
    sft_dataset_dir,
    lang_select,
    sft_format_status,
    sft_dataset_state,
    sft_start_btn,
    sft_epochs,
    sft_learning_rate,
    sft_batch_size,
    sft_max_seq_length,
    sft_gradient_accumulation,
    sft_warmup_ratio,
    sft_system_prompt,
    sft_progress,
    sft_log,
    sft_plot,
    sft_stop_btn,
):
    sft_refresh_model_btn.click(
        fn=sft_refresh_models,
        inputs=[],
        outputs=[sft_base_model],
    )

    sft_validate_btn.click(
        fn=sft_load_dataset,
        inputs=[sft_dataset_file, sft_dataset_dir, lang_select],
        outputs=[sft_format_status, sft_dataset_state],
    )

    sft_start_btn.click(
        fn=sft_train_cb,
        inputs=[
            sft_base_model,
            sft_dataset_state,
            sft_epochs,
            sft_learning_rate,
            sft_batch_size,
            sft_max_seq_length,
            sft_gradient_accumulation,
            sft_warmup_ratio,
            sft_system_prompt,
            lang_select,
        ],
        outputs=[sft_progress, sft_log, sft_plot],
    )

    sft_stop_btn.click(
        fn=sft_stop_cb,
        inputs=[],
        outputs=[sft_log],
    )
