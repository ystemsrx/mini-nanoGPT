from src.ui.callbacks.comparison import select_model_for_comparison_cb, dual_inference_cb


def bind_comparison(
    comp_left_model,
    comp_right_model,
    comp_left_params,
    comp_left_plot,
    comp_left_history,
    comp_left_data_dir,
    comp_left_out_dir,
    comp_right_params,
    comp_right_plot,
    comp_right_history,
    comp_right_data_dir,
    comp_right_out_dir,
    comp_generate_btn,
    comp_prompt,
    comp_left_num_samples,
    comp_left_max_tokens,
    comp_left_temperature,
    comp_left_top_k,
    comp_left_dtype,
    comp_left_seed,
    comp_right_num_samples,
    comp_right_max_tokens,
    comp_right_temperature,
    comp_right_top_k,
    comp_right_dtype,
    comp_right_seed,
    comp_left_output,
    comp_right_output,
):
    comp_left_model.change(
        fn=lambda sel: select_model_for_comparison_cb(sel, True),
        inputs=[comp_left_model],
        outputs=[comp_left_params, comp_left_plot, comp_left_history, comp_left_data_dir, comp_left_out_dir],
    )

    comp_right_model.change(
        fn=lambda sel: select_model_for_comparison_cb(sel, False),
        inputs=[comp_right_model],
        outputs=[comp_right_params, comp_right_plot, comp_right_history, comp_right_data_dir, comp_right_out_dir],
    )

    # Connect the generate button to the dual inference callback
    comp_generate_btn.click(
        fn=dual_inference_cb,
        inputs=[
            comp_left_data_dir,
            comp_left_out_dir,
            comp_right_data_dir,
            comp_right_out_dir,
            comp_prompt,
            comp_left_num_samples,
            comp_left_max_tokens,
            comp_left_temperature,
            comp_left_top_k,
            comp_left_dtype,
            comp_left_seed,
            comp_right_num_samples,
            comp_right_max_tokens,
            comp_right_temperature,
            comp_right_top_k,
            comp_right_dtype,
            comp_right_seed,
        ],
        outputs=[comp_left_output, comp_right_output],
    )
