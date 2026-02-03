from src.ui.callbacks.data_processing import data_processing_cb


def bind_data_processing(
    process_btn,
    new_model_chk,
    model_name_box,
    model_dropdown,
    input_text,
    txt_dir,
    train_split,
    no_val_set,
    use_custom_tokenizer,
    num_proc,
    lang_select,
    process_output,
    comp_left_model,
    comp_right_model,
):
    process_btn.click(
        fn=data_processing_cb,
        inputs=[
            new_model_chk,
            model_name_box,
            model_dropdown,
            input_text,
            txt_dir,
            train_split,
            no_val_set,
            use_custom_tokenizer,
            num_proc,
            lang_select,
        ],
        outputs=[process_output, model_dropdown, comp_left_model, comp_right_model],
    )
