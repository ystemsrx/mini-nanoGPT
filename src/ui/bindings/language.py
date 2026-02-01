from src.ui.callbacks.language import switch_language


def bind_language(
    lang_select,
    lang_select_outputs,
):
    lang_select.change(
        fn=switch_language,
        inputs=[lang_select],
        outputs=lang_select_outputs,
    )
