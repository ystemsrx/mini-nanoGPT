import os

import torch
import gradio as gr

from src.config import LANG_JSON
from src.sft import validate_alpaca_format, load_sft_dataset, sft_train_generator, stop_sft_training
from src.infer_cache import ModelCache
from src.ui.charts import generate_loss_chart_html, make_progress_html
from src.ui.helpers import _get_model_choices_list
from src.ui.state import dbm


def sft_refresh_models():
    choices = _get_model_choices_list()
    return gr.update(choices=choices, value=None)


def sft_load_dataset(file_obj, dir_path, lang_code):
    T_current = LANG_JSON[lang_code]

    # Reset status
    msg = T_current["sft_no_dataset"]
    dataset = []

    # Prioritize file upload
    if file_obj is not None:
        dataset, msg = load_sft_dataset(file_path=file_obj.name)
    elif dir_path and dir_path.strip():
        dataset, msg = load_sft_dataset(dir_path=dir_path)

    is_valid, _ = validate_alpaca_format(dataset)
    status_val = T_current["sft_valid_format"] if is_valid else f"{T_current['sft_invalid_format']}: {msg}"

    return status_val, dataset


def sft_train_cb(
    model_selection, dataset, epochs, lr, batch_size,
    max_seq_len, grad_acc, warmup_ratio, system_prompt,
    lang_code,
):
    T_current = LANG_JSON[lang_code]

    # Pre-SFT training cleanup: Clear any residual GPU resources from previous failed training
    try:
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cache = ModelCache()
        cache.clear_cache()
        print("🧹 Pre-SFT training cleanup completed")
    except Exception as cleanup_err:
        print(f"Warning: Pre-SFT training cleanup encountered an issue: {cleanup_err}")

    empty_plot = generate_loss_chart_html([], [])

    if not model_selection or " - " not in model_selection:
        yield f"<div style='color:red;'>❌ Please select a base model</div>", "", empty_plot
        return

    model_id = int(model_selection.split(" - ")[0])
    model_info = dbm.get_model(model_id)
    if not model_info:
        yield f"<div style='color:red;'>❌ Model not found</div>", "", empty_plot
        return

    base_ckpt_path = os.path.join(model_info["out_dir"], "ckpt.pt")

    if not dataset:
        yield f"<div style='color:red;'>{T_current['sft_no_dataset']}</div>", "", empty_plot
        return

    # Create SFT output directory
    sft_out_dir = os.path.join(model_info["out_dir"], "sft")
    os.makedirs(sft_out_dir, exist_ok=True)

    yield make_progress_html(0, 100), "🚀 Starting SFT Training...", empty_plot

    try:
        generator = sft_train_generator(
            base_model_ckpt_path=base_ckpt_path,
            data_dir=model_info["processed_data_dir"],
            dataset=dataset,
            out_dir=sft_out_dir,
            epochs=int(epochs),
            learning_rate=lr,
            batch_size=int(batch_size),
            max_seq_length=int(max_seq_len),
            gradient_accumulation_steps=int(grad_acc),
            warmup_ratio=warmup_ratio,
            system_prompt=system_prompt,
        )

        for progress_html, log_msg, plot_data in generator:
            # Plot data format: (steps, losses, val_steps, val_losses)
            if plot_data and len(plot_data) >= 2:
                train_data = list(zip(plot_data[0], plot_data[1]))
                val_data = list(zip(plot_data[2], plot_data[3])) if len(plot_data) > 3 else []
                plot_html = generate_loss_chart_html(train_data, val_data)
            else:
                plot_html = empty_plot

            yield progress_html, log_msg, plot_html
    except Exception as e:
        import traceback

        print(f"SFT Training callback error: {traceback.format_exc()}")
        err_msg = f"Runtime Error in SFT Training: {str(e)}"

        # Critical: Clean up GPU resources after SFT training failure
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            try:
                cache = ModelCache()
                cache.clear_cache()
            except Exception as cache_err:
                print(f"Warning: Failed to clear model cache: {cache_err}")
            print("🧹 Post-SFT-error cleanup completed - GPU resources released")
        except Exception as cleanup_err:
            print(f"Warning: Post-SFT-error cleanup failed: {cleanup_err}")

        yield f"<div style='color:red;'>{err_msg}</div>", "", empty_plot


def sft_stop_cb():
    stop_sft_training()
    return "🛑 Stopping SFT..."
