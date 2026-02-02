import gradio as gr

from src.ui.callbacks.comparison import (
    select_model_for_comparison_cb,
    dual_inference_cb,
    stop_comparison_cb,
    dual_chat_cb,
    clear_compare_chat_cb,
)


def bind_comparison(
    comp_left_model,
    comp_right_model,
    comp_left_params,
    comp_left_plot,
    comp_left_history,
    comp_left_data_dir,
    comp_left_out_dir,
    comp_left_has_sft,
    comp_right_params,
    comp_right_plot,
    comp_right_history,
    comp_right_data_dir,
    comp_right_out_dir,
    comp_right_has_sft,
    comp_chat_mode,
    comp_standard_group,
    comp_chat_group,
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
    comp_stop_btn,
    comp_system_prompt,
    comp_user_input,
    comp_send_btn,
    comp_clear_btn,
    comp_left_chatbot,
    comp_right_chatbot,
):
    comp_left_model.change(
        fn=lambda sel: select_model_for_comparison_cb(sel, True),
        inputs=[comp_left_model],
        outputs=[
            comp_left_params,
            comp_left_plot,
            comp_left_history,
            comp_left_data_dir,
            comp_left_out_dir,
            comp_left_has_sft,
        ],
    )

    comp_right_model.change(
        fn=lambda sel: select_model_for_comparison_cb(sel, False),
        inputs=[comp_right_model],
        outputs=[
            comp_right_params,
            comp_right_plot,
            comp_right_history,
            comp_right_data_dir,
            comp_right_out_dir,
            comp_right_has_sft,
        ],
    )

    def update_comp_chat_mode(left_has_sft, right_has_sft, current_value):
        enabled = bool(left_has_sft and right_has_sft)
        if not enabled:
            return (
                gr.update(value=False, interactive=False),
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=True),
                gr.update(visible=True),
            )
        return (
            gr.update(value=bool(current_value), interactive=True),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
        )

    comp_left_has_sft.change(
        fn=update_comp_chat_mode,
        inputs=[comp_left_has_sft, comp_right_has_sft, comp_chat_mode],
        outputs=[comp_chat_mode, comp_chat_group, comp_standard_group, comp_left_num_samples, comp_right_num_samples],
    )

    comp_right_has_sft.change(
        fn=update_comp_chat_mode,
        inputs=[comp_left_has_sft, comp_right_has_sft, comp_chat_mode],
        outputs=[comp_chat_mode, comp_chat_group, comp_standard_group, comp_left_num_samples, comp_right_num_samples],
    )

    def toggle_comp_chat_mode(is_chat):
        return {
            comp_chat_group: gr.update(visible=is_chat),
            comp_standard_group: gr.update(visible=not is_chat),
            comp_left_num_samples: gr.update(visible=not is_chat),
            comp_right_num_samples: gr.update(visible=not is_chat),
        }

    comp_chat_mode.change(
        fn=toggle_comp_chat_mode,
        inputs=[comp_chat_mode],
        outputs=[comp_chat_group, comp_standard_group, comp_left_num_samples, comp_right_num_samples],
    )

    # Connect the generate button to the dual inference callback
    comp_generate_btn.click(
        fn=dual_inference_cb,
        inputs=[
            comp_left_data_dir,
            comp_left_out_dir,
            comp_right_data_dir,
            comp_right_out_dir,
            comp_left_model,
            comp_right_model,
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
        outputs=[comp_left_output, comp_right_output, comp_generate_btn, comp_stop_btn],
    )

    comp_stop_btn.click(
        fn=stop_comparison_cb,
        inputs=[],
        outputs=[comp_left_output, comp_right_output, comp_generate_btn, comp_stop_btn],
    )

    comp_send_btn.click(
        fn=dual_chat_cb,
        inputs=[
            comp_user_input,
            comp_left_chatbot,
            comp_right_chatbot,
            comp_left_model,
            comp_right_model,
            comp_system_prompt,
            comp_left_max_tokens,
            comp_left_temperature,
            comp_left_top_k,
            comp_left_seed,
            comp_right_max_tokens,
            comp_right_temperature,
            comp_right_top_k,
            comp_right_seed,
        ],
        outputs=[comp_user_input, comp_left_chatbot, comp_right_chatbot],
    )

    comp_user_input.submit(
        fn=dual_chat_cb,
        inputs=[
            comp_user_input,
            comp_left_chatbot,
            comp_right_chatbot,
            comp_left_model,
            comp_right_model,
            comp_system_prompt,
            comp_left_max_tokens,
            comp_left_temperature,
            comp_left_top_k,
            comp_left_seed,
            comp_right_max_tokens,
            comp_right_temperature,
            comp_right_top_k,
            comp_right_seed,
        ],
        outputs=[comp_user_input, comp_left_chatbot, comp_right_chatbot],
    )

    comp_clear_btn.click(
        fn=clear_compare_chat_cb,
        inputs=[],
        outputs=[comp_user_input, comp_left_chatbot, comp_right_chatbot],
    )
