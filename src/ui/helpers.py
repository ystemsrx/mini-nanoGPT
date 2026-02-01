import os
import pickle

from .charts import generate_loss_chart_html
from .state import dbm


def _match_device_value(config_device: str, available_devices: list) -> str:
    """
    Match configuration device value with available devices

    Args:
        config_device: Device from config (e.g., 'cuda', 'cpu')
        available_devices: List of actual available devices (e.g., ['cuda:0', 'cpu'])

    Returns:
        Matched device name from available_devices
    """
    if config_device in available_devices:
        return config_device

    # If config has 'cuda' but available devices have 'cuda:0', etc.
    if config_device == "cuda":
        for device in available_devices:
            if device.startswith("cuda:"):
                return device

    # If config has 'cuda:X' but that specific device is not available,
    # fall back to first available CUDA device
    if config_device.startswith("cuda:"):
        for device in available_devices:
            if device.startswith("cuda:"):
                return device

    # Default fallback to CPU if available, or first device in list
    if "cpu" in available_devices:
        return "cpu"

    return available_devices[0] if available_devices else "cpu"


def _get_model_choices_list():
    return [f"{m['id']} - {m['name']}" for m in dbm.get_all_models()]


def _create_plot_html_from_log(loss_log_path: str):
    if not (loss_log_path and os.path.exists(loss_log_path)):
        return generate_loss_chart_html([], [])
    try:
        with open(loss_log_path, "rb") as f:
            loss_dict = pickle.load(f)

        tr_steps = loss_dict.get("train_plot_steps", [])
        tr_losses = loss_dict.get("train_plot_losses", [])
        val_steps = loss_dict.get("val_plot_steps", [])
        val_losses = loss_dict.get("val_plot_losses", [])

        train_data = []
        if tr_steps and tr_losses:
            train_data = list(zip(tr_steps, tr_losses))

        val_data = []
        if val_steps and val_losses:
            val_data = list(zip(val_steps, val_losses))

        return generate_loss_chart_html(train_data, val_data)
    except Exception as e:
        print(f"Error in _create_plot_html_from_log: {str(e)}")
        return generate_loss_chart_html([], [])
