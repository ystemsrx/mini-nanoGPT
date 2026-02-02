import gradio as gr

from src.ui.callbacks.model_management import select_model_cb, delete_model_cb
from src.ui.helpers import _get_model_choices_list


def bind_model_management(
    demo,
    model_dropdown,
    delete_model_btn,
    refresh_models_btn,
    outputs_for_model_select_and_delete,
    comp_left_model,
    comp_right_model,
):
    model_dropdown.change(
        fn=select_model_cb,
        inputs=[model_dropdown],
        outputs=outputs_for_model_select_and_delete,
    )

    model_dropdown.change(
        fn=lambda sel: gr.update(interactive=bool(sel)),
        inputs=[model_dropdown],
        outputs=[delete_model_btn],
    )

    delete_model_btn.click(
        fn=delete_model_cb,
        inputs=[model_dropdown],
        outputs=[model_dropdown, comp_left_model, comp_right_model] + outputs_for_model_select_and_delete,
    )

    refresh_models_btn.click(
        lambda: [gr.update(choices=_get_model_choices_list()) for _ in range(3)],
        [],
        [model_dropdown, comp_left_model, comp_right_model],
    )

    # Refresh model list on page load to ensure newly created models are displayed
    demo.load(
        fn=lambda: [gr.update(choices=_get_model_choices_list()) for _ in range(3)],
        inputs=None,
        outputs=[model_dropdown, comp_left_model, comp_right_model],
        queue=False,
    )
