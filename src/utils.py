# src/utils.py

import os

def compose_model_dirs(model_name: str, model_id: int):
    """
    Generate model directories:
        · raw_data_dir   : ./data/{model_name}_{model_id}/raw/
        · processed_dir  : ./data/{model_name}_{model_id}/processed/
        · out_dir        : ./out/{model_name}_{model_id}/
    Returns a tuple of strings: (raw_data_dir, processed_data_dir, out_dir)
    """
    folder = f"{model_name}_{model_id}"
    raw_data_dir = os.path.join("data", folder, "raw")
    processed_data_dir = os.path.join("data", folder, "processed")
    out_dir = os.path.join("out", folder)
    return raw_data_dir, processed_data_dir, out_dir
