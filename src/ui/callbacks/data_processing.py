import gradio as gr

from src.config import LANG_JSON
from src.data_process import process_data
from src.ui.helpers import _get_model_choices_list


def data_processing_cb(
    new_flag, model_name, dropdown_val,
    txt, ddir,
    sp, no_val, use_custom_tokenizer, num_proc_,
    lang_code,
):
    try:
        T_current = LANG_JSON[lang_code]

        sel_id = int(dropdown_val.split(" - ")[0]) if dropdown_val and " - " in dropdown_val else None
        info = process_data(
            model_name=model_name.strip() or "unnamed",
            new_model=new_flag,
            selected_model_id=sel_id,
            input_text=txt,
            input_dir=ddir,
            train_split_ratio=sp,
            no_validation=no_val,
            use_custom_tokenizer=use_custom_tokenizer,
            num_proc=int(num_proc_),
        )
        new_choices = _get_model_choices_list()
        new_val = f"{info['model_id']} - {model_name.strip() or 'unnamed'}"
        msg = (
            f"✅ {T_current['dp_result']}:\n"
            f"model_id = {info['model_id']}\n"
            f"processed_dir = {info['processed_data_dir']}\n"
            f"vocab_size = {info['vocab_size']}\n"
            f"tokenizer = {info['tokenizer']}\n"
            f"train_size = {info['train_size']}" +
            (f"\nval_size = {info['val_size']}" if 'val_size' in info else "\n(no val)")
        )
        # Update main dropdown and comparison dropdowns
        return msg, gr.update(choices=new_choices, value=new_val), gr.update(choices=new_choices), gr.update(choices=new_choices)
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(), gr.update(), gr.update()
