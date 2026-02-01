import gradio as gr

from src.ui.callbacks.chat import chat_cb, clear_chat
from src.ui.callbacks.inference import inference_cb


def bind_inference(
    inf_chat_mode,
    chat_interface_group,
    standard_interface_group,
    prompt_box,
    inf_btn,
    data_dir_inf,
    out_dir_inf,
    num_samples_box,
    max_new_tokens_box,
    temperature_box,
    top_k_box,
    dtype_box_inf,
    device_box_inf,
    seed_box_inf,
    inf_output,
    inf_advanced_output,
    inf_send_btn,
    inf_user_input,
    chatbot,
    model_dropdown,
    inf_system_prompt,
    chat_advanced_output,
    inf_clear_btn,
):
    # Visibility Logic for chat mode
    def toggle_chat_mode(is_chat):
        return {
            chat_interface_group: gr.update(visible=is_chat),
            standard_interface_group: gr.update(visible=not is_chat),
            prompt_box: gr.update(visible=not is_chat),
        }

    inf_chat_mode.change(
        fn=toggle_chat_mode,
        inputs=[inf_chat_mode],
        outputs=[chat_interface_group, standard_interface_group, prompt_box],
    )

    # Single model inference
    inf_btn.click(
        fn=inference_cb,
        inputs=[
            data_dir_inf,
            out_dir_inf,
            prompt_box,
            num_samples_box,
            max_new_tokens_box,
            temperature_box,
            top_k_box,
            dtype_box_inf,
            device_box_inf,
            seed_box_inf,
        ],
        outputs=[inf_output, inf_advanced_output],
    )

    # Chat callbacks
    inf_send_btn.click(
        fn=chat_cb,
        inputs=[
            inf_user_input,
            chatbot,
            model_dropdown,
            inf_system_prompt,
            max_new_tokens_box,
            temperature_box,
            top_k_box,
            seed_box_inf,
            device_box_inf,
        ],
        outputs=[inf_user_input, chatbot, chat_advanced_output],
    )

    inf_user_input.submit(
        fn=chat_cb,
        inputs=[
            inf_user_input,
            chatbot,
            model_dropdown,
            inf_system_prompt,
            max_new_tokens_box,
            temperature_box,
            top_k_box,
            seed_box_inf,
            device_box_inf,
        ],
        outputs=[inf_user_input, chatbot, chat_advanced_output],
    )

    inf_clear_btn.click(
        fn=clear_chat,
        inputs=[model_dropdown],
        outputs=[chatbot, chat_advanced_output],
    )

    # Also handle the built-in clear button (trash icon) in the Chatbot component
    # This ensures database history is also cleared when user clicks the trash icon
    chatbot.clear(
        fn=clear_chat,
        inputs=[model_dropdown],
        outputs=[chatbot, chat_advanced_output],
    )