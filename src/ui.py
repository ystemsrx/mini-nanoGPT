# src/ui.py
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import torch._dynamo
import time
torch._dynamo.config.suppress_errors = True

import gradio as gr

from src.config import DEFAULT_CONFIG, LANG_JSON
from src.db_manager import DBManager
from src.data_process import process_data
from src.train import train_model_generator, stop_training
from src.sft import (
    validate_alpaca_format, load_sft_dataset, sft_train_generator,
    stop_sft_training, chat_generate, tokenize_user_input
)
from src.infer_cache import cached_generate_text, ModelCache, UnknownTokenError
from src.device_manager import device_manager
from src.gpt_model import GPTConfig, GPT
from src.gpt_self_attn import GPTSelfAttnConfig, GPTSelfAttn

import queue
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

dbm = DBManager()

# --- SVG Chart Generation Function ---
def generate_loss_chart_html(
    train_data, # List of (epoch, loss) tuples
    val_data,   # List of (epoch, loss) tuples
):
    svg_width = 800
    svg_height = 400
    margin_left = 60 # For Y-axis labels
    margin_top = 50
    margin_bottom = 60 # For X-axis title
    margin_right = 30
    chart_width = svg_width - margin_left - margin_right
    chart_height = svg_height - margin_top - margin_bottom
    axis_base_y = margin_top + chart_height  # Y-axis bottom (loss=0)

    # --- Determine Dynamic Axis Ranges ---
    all_epochs = [d[0] for d in train_data + val_data if d]
    all_losses = [d[1] for d in train_data + val_data if d]

    current_max_data_epoch = max(all_epochs) if all_epochs else 0
    if not all_epochs:
        display_max_epoch = 10.0
    elif current_max_data_epoch < 10:
        display_max_epoch = 10.0
    else:
        display_max_epoch = float(current_max_data_epoch)

    display_min_epoch = 0.0

    if not all_losses:
        data_min_loss = 0.0
        data_max_loss = 1.0
    else:
        data_min_loss = min(all_losses) if all_losses else 0.0 # Ensure min isn't called on empty
        data_max_loss = max(all_losses) if all_losses else 1.0 # Ensure max isn't called on empty


    loss_range = data_max_loss - data_min_loss
    if loss_range < 0.01: # Handle very flat data
        y_axis_min_display = max(0.0, data_min_loss - 0.05)
        y_axis_max_display = y_axis_min_display + 0.1
    else:
        y_axis_min_display = max(0.0, data_min_loss - loss_range * 0.15)
        y_axis_max_display = data_max_loss + loss_range * 0.15
    
    if y_axis_min_display >= y_axis_max_display: # Ensure max > min
        y_axis_max_display = y_axis_min_display + 0.1


    def to_svg_coords(epoch, loss):
        if display_max_epoch == display_min_epoch: # Avoid division by zero
            x_scaled = 0
        else:
            x_scaled = (epoch - display_min_epoch) / (display_max_epoch - display_min_epoch)
        x = margin_left + x_scaled * chart_width

        y_display_range = y_axis_max_display - y_axis_min_display
        if y_display_range == 0: # Avoid division by zero
            y_scaled = 0
        else:
            y_scaled = (loss - y_axis_min_display) / y_display_range
        
        y = margin_top + chart_height - y_scaled * chart_height # Invert Y for SVG

        # Clamp coordinates to chart area
        x = max(margin_left, min(x, margin_left + chart_width))
        y = max(margin_top, min(y, margin_top + chart_height))
        return round(x, 2), round(y, 2)

    # --- Generate SVG elements for data ---
    def create_path_elements(data_points, css_class_prefix):
        path_static_d = ""
        path_anim_d = ""
        anim_segment_length = 0
        circles_svg = ""
        area_d = ""

        svg_points = [to_svg_coords(e, l) for e, l in data_points if e is not None and l is not None]


        if svg_points:
            # Generate circle points
            for i, (x, y) in enumerate(svg_points):
                css_class = f"point-{css_class_prefix}"
                if i == len(svg_points) - 1:
                    css_class += " point-new"
                circles_svg += f'<circle class="{css_class}" cx="{x}" cy="{y}" r="4"></circle>\n'
            
            _catmull_rom_tension = 1/6 

            if len(svg_points) == 1:
                p0 = svg_points[0]
                path_static_d = f"M {p0[0]},{p0[1]}" 
                area_d = f"M {p0[0]},{p0[1]} L {p0[0]},{axis_base_y} L {p0[0]},{axis_base_y} Z"

            elif len(svg_points) >= 2:
                control_points_list = []
                for i in range(len(svg_points) - 1): 
                    p_i = svg_points[i]
                    p_i_plus_1 = svg_points[i+1]
                    
                    p_i_minus_1 = svg_points[i-1] if i > 0 else p_i 
                    k1_x = p_i[0] + (p_i_plus_1[0] - p_i_minus_1[0]) * _catmull_rom_tension
                    k1_y = p_i[1] + (p_i_plus_1[1] - p_i_minus_1[1]) * _catmull_rom_tension

                    p_i_plus_2 = svg_points[i+2] if i + 2 < len(svg_points) else p_i_plus_1
                    k2_x = p_i_plus_1[0] - (p_i_plus_2[0] - p_i[0]) * _catmull_rom_tension
                    k2_y = p_i_plus_1[1] - (p_i_plus_2[1] - p_i[1]) * _catmull_rom_tension
                    control_points_list.append((k1_x, k1_y, k2_x, k2_y))

                full_smooth_path_for_area = f"M {svg_points[0][0]},{svg_points[0][1]}"
                for i in range(len(svg_points) - 1):
                    k1x, k1y, k2x, k2y = control_points_list[i]
                    p_next = svg_points[i+1]
                    full_smooth_path_for_area += f" C {round(k1x,2)},{round(k1y,2)} {round(k2x,2)},{round(k2y,2)} {p_next[0]},{p_next[1]}"
                
                area_d = full_smooth_path_for_area + f" L {svg_points[-1][0]},{axis_base_y} L {svg_points[0][0]},{axis_base_y} Z"

                if len(svg_points) == 2: 
                    path_static_d = f"M {svg_points[0][0]},{svg_points[0][1]}" 
                    k1x, k1y, k2x, k2y = control_points_list[0]
                    path_anim_d = f"M {svg_points[0][0]},{svg_points[0][1]} C {round(k1x,2)},{round(k1y,2)} {round(k2x,2)},{round(k2y,2)} {svg_points[1][0]},{svg_points[1][1]}"
                else: 
                    path_static_d = f"M {svg_points[0][0]},{svg_points[0][1]}"
                    for i in range(len(svg_points) - 2): 
                        k1x, k1y, k2x, k2y = control_points_list[i]
                        p_next = svg_points[i+1]
                        path_static_d += f" C {round(k1x,2)},{round(k1y,2)} {round(k2x,2)},{round(k2y,2)} {p_next[0]},{p_next[1]}"
                    
                    k1x_anim, k1y_anim, k2x_anim, k2y_anim = control_points_list[-1] 
                    p_prev_anim = svg_points[-2]
                    p_curr_anim = svg_points[-1]
                    path_anim_d = f"M {p_prev_anim[0]},{p_prev_anim[1]} C {round(k1x_anim,2)},{round(k1y_anim,2)} {round(k2x_anim,2)},{round(k2y_anim,2)} {p_curr_anim[0]},{p_curr_anim[1]}"
                
                if len(svg_points) >=2: # Ensure there are at least two points for segment length
                    p_prev_for_len = svg_points[-2]
                    p_curr_for_len = svg_points[-1]
                    anim_segment_length = np.sqrt((p_curr_for_len[0] - p_prev_for_len[0])**2 + (p_curr_for_len[1] - p_prev_for_len[1])**2)
                    anim_segment_length = max(0.1, round(anim_segment_length, 2)) 
                else: # Should not happen if len(svg_points) >= 2, but as a fallback
                    anim_segment_length = 0.1


        return path_static_d, path_anim_d, circles_svg, area_d, anim_segment_length

    train_path_static_d, train_path_anim_d, train_circles_svg, train_area_d, train_anim_segment_length = create_path_elements(train_data, "train")
    val_path_static_d, val_path_anim_d, val_circles_svg, val_area_d, val_anim_segment_length = create_path_elements(val_data, "val")
    
    x_axis_labels_svg = ""
    num_x_ticks = 5 
    effective_display_max_epoch = max(display_min_epoch, display_max_epoch)
    x_tick_values = np.linspace(display_min_epoch, effective_display_max_epoch, num_x_ticks + 1)
    if effective_display_max_epoch == display_min_epoch and effective_display_max_epoch == 0: # Handles case of single point at 0 or no data
         x_tick_values = np.linspace(0, 10, num_x_ticks + 1) # Default axis if no data
    elif effective_display_max_epoch == display_min_epoch : # Single point not at 0
        x_tick_values = [display_min_epoch]


    for epoch_val in x_tick_values:
        x_coord, _ = to_svg_coords(epoch_val, y_axis_min_display) 
        label = f"{epoch_val:.1f}" if effective_display_max_epoch < 10 and effective_display_max_epoch != 0 else f"{int(round(epoch_val))}"
        x_axis_labels_svg += f'<text class="axis-label" x="{x_coord}" y="{axis_base_y + 25}">{label}</text>\n'
    x_axis_labels_svg += f'<text class="axis-title" x="{margin_left + chart_width / 2}" y="{axis_base_y + 45}">Steps</text>\n' # Changed from Epoch to Steps

    y_axis_labels_svg = ""
    num_y_ticks = 5 
    y_tick_values = np.linspace(y_axis_min_display, y_axis_max_display, num_y_ticks + 1)
    if y_axis_min_display == y_axis_max_display: 
        y_tick_values = [y_axis_min_display] if y_axis_min_display != 0 else np.linspace(0,1, num_y_ticks+1)


    for loss_val in y_tick_values:
        _, y_coord = to_svg_coords(display_min_epoch, loss_val) 
        y_axis_labels_svg += f'<text class="y-axis-label" x="{margin_left - 10}" y="{y_coord}">{loss_val:.2f}</text>\n'
    y_axis_labels_svg += f"""
        <text class="axis-title" 
              transform="rotate(-90 {margin_left - 45} {margin_top + chart_height/2})" 
              x="{margin_left - 45}" y="{margin_top + chart_height/2}">
              Loss Rate
        </text>
    """

    grid_lines_svg = ""
    for loss_val in y_tick_values:
        _, y_coord = to_svg_coords(display_min_epoch, loss_val)  
        grid_lines_svg += f'<line class="grid-line" x1="{margin_left}" y1="{y_coord}" x2="{margin_left + chart_width}" y2="{y_coord}" />\n'
    
    if len(y_tick_values) > 1:
        for i in range(len(y_tick_values) - 1):
            main_y_start = y_tick_values[i]
            main_y_end = y_tick_values[i+1]
            y_interval = (main_y_end - main_y_start) / 6  
            for j in range(1, 6):  
                intermediate_y = main_y_start + j * y_interval
                _, y_coord = to_svg_coords(display_min_epoch, intermediate_y)
                grid_lines_svg += f'<line class="grid-line grid-line-dashed" x1="{margin_left}" y1="{y_coord}" x2="{margin_left + chart_width}" y2="{y_coord}" />\n'


    train_anim_segment_svg = ""
    if train_path_anim_d and train_anim_segment_length > 0:
        train_anim_segment_svg = f'<path class="line-segment-anim line-train-segment-anim" d="{train_path_anim_d}" style="stroke-dasharray: {train_anim_segment_length}; stroke-dashoffset: {train_anim_segment_length};" />'

    val_anim_segment_svg = ""
    if val_path_anim_d and val_anim_segment_length > 0:
        val_anim_segment_svg = f'<path class="line-segment-anim line-val-segment-anim" d="{val_path_anim_d}" style="stroke-dasharray: {val_anim_segment_length}; stroke-dashoffset: {val_anim_segment_length};" />'

    legend_items_html = ""
    if train_data:
        legend_items_html += """
        <div class="legend-item">
            <span class="legend-color legend-color-train"></span>
            <span>Training Loss</span>
        </div>"""
    if val_data:
        legend_items_html += """
        <div class="legend-item">
            <span class="legend-color legend-color-val"></span>
            <span>Validation Loss</span>
        </div>"""

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Loss Rate Chart</title>
<style>
  :root {{
    --accent:#ffbb00; 
    --accent-light:rgba(255, 201, 93, 0.3); 
    --train:#3b82f6; 
    --train-light:rgba(59, 130, 246, 0.15); 
    --grid:#e0e0e0; 
    --axis-line-color:#888; 
    --label-color:#666; 
    --title-color:#333;
    --bg-grid:#fdfdfd; 
    --point-new-glow: rgba(255, 100, 100, 0.8); 
  }}
  * {{box-sizing:border-box;margin:0;padding:0;}}
  .chart-container {{ 
    width:100%; 
    max-width:{svg_width}px; 
    background:#fff; 
    border-radius:12px;box-shadow:0 6px 20px rgba(0,0,0,.07);
    padding:20px; 
    margin: auto; 
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  }}
  h2.chart-main-title {{text-align:center;color:var(--title-color);margin-bottom:18px; font-weight: 600;}}
  .chart-wrapper {{ position: relative; overflow: hidden; }}
  svg {{width:100%;height:auto;overflow:visible;}}
  .chart-bg {{ fill: var(--bg-grid); stroke: var(--grid); stroke-width:0.5; }}
  .grid-line {{ stroke: var(--grid); stroke-width: 1; }}
  .grid-line-dashed {{ stroke-dasharray: 2 2; }}
  .axis-line {{ stroke: var(--axis-line-color); stroke-width: 1.5; }}
  .axis-label {{ font-size: 11px; fill: var(--label-color); text-anchor: middle; dominant-baseline: hanging;}}
  .y-axis-label {{ font-size: 11px; fill: var(--label-color); text-anchor: end; dominant-baseline: middle;}}
  .axis-title {{ font-size: 13px; fill: var(--title-color); text-anchor: middle; font-weight: 500;}}

  .line-base {{ 
    fill: none; 
    stroke-width: 2.5; 
    stroke-linecap: round; 
    stroke-linejoin: round; 
  }}
  .line-val-base {{ stroke: var(--accent); }}
  .line-train-base {{ stroke: var(--train); }}

  .line-segment-anim {{ 
    fill: none; 
    stroke-width: 2.5; 
    stroke-linecap: round; 
    stroke-linejoin: round; 
    animation: drawLineSegment 0.5s cubic-bezier(0.25, 0.1, 0.25, 1) forwards; 
  }}
  .line-val-segment-anim {{ stroke: var(--accent); }}
  .line-train-segment-anim {{ stroke: var(--train); }}

  .area-val {{ 
    fill:url(#gradient-val); 
    opacity:0.8; 
  }}
  .area-train {{ 
    fill:url(#gradient-train); 
    opacity:0.8; 
  }}

  .point-val, .point-train {{ 
    fill:#fff; stroke-width:1.5; transition:r .2s ease-out; 
  }}
  .point-val {{ stroke:var(--accent); }}
  .point-train {{ stroke:var(--train); }}
  .point-val:hover, .point-train:hover {{r:6;}}

  .point-new {{ 
    r: 0;  
    animation: scaleInPoint 0.4s 0.1s cubic-bezier(0.34, 1.56, 0.64, 1) forwards; 
  }}
  .point-new.point-train {{ fill: var(--train); }}
  .point-new.point-val {{ fill: var(--accent); }}

  .legend {{ display: flex; justify-content: center; margin-top: 20px; gap: 20px; flex-wrap: wrap;}}
  .legend-item {{ display: flex; align-items: center; font-size: 13px; color: var(--label-color); }}
  .legend-color {{ width: 14px; height: 14px; border-radius: 3px; margin-right: 7px; }}
  .legend-color-train {{ background-color: var(--train); }}
  .legend-color-val {{ background-color: var(--accent); }}

  @keyframes drawLineSegment {{ 
    to {{stroke-dashoffset:0;}} 
  }}
  @keyframes scaleInPoint {{ 
    0% {{ r: 0; opacity: 0; }} 
    70% {{ r: 5.5; opacity: 1; }} 
    100% {{ r: 4; opacity: 1; }} 
  }}
</style>
</head>
<body>
  <div class="chart-container">
    <h2 class="chart-main-title">Loss Curve</h2>
    <div class="chart-wrapper">
      <svg viewBox="0 0 {svg_width} {svg_height}" role="img" aria-labelledby="chartTitle">
        <title id="chartTitle">Training and Validation Loss Rate Over Steps</title>
        <defs>
          <linearGradient id="gradient-val" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="var(--accent-light)" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="var(--accent-light)" stop-opacity="0.05"/>
          </linearGradient>
          <linearGradient id="gradient-train" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stop-color="var(--train-light)" stop-opacity="0.7"/>
            <stop offset="100%" stop-color="var(--train-light)" stop-opacity="0.05"/>
          </linearGradient>
        </defs>
        
        <rect class="chart-bg" x="{margin_left}" y="{margin_top}" width="{chart_width}" height="{chart_height}"></rect>
        
        {grid_lines_svg}
        
        <line class="axis-line" x1="{margin_left}" y1="{axis_base_y}" x2="{margin_left + chart_width}" y2="{axis_base_y}" />
        <line class="axis-line" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_base_y}" />
        {x_axis_labels_svg}
        {y_axis_labels_svg}

        {f'<path class="area-val" d="{val_area_d}" />' if val_area_d else ""}
        {f'<path class="area-train" d="{train_area_d}" />' if train_area_d else ""}

        {f'<path class="line-base line-val-base" d="{val_path_static_d}" />' if val_path_static_d else ""}
        {f'<path class="line-base line-train-base" d="{train_path_static_d}" />' if train_path_static_d else ""}
        
        {val_anim_segment_svg}
        {train_anim_segment_svg}
        
        <g>{val_circles_svg if val_data else ""}</g>
        <g>{train_circles_svg if train_data else ""}</g>
      </svg>
    </div>
    
    <div class="legend">
      {legend_items_html}
    </div>
  </div>
</body>
</html>
"""
    return html_content

def make_progress_html(progress_val, max_val, color='blue'):
    """Generate HTML for a progress bar."""
    return (
        f"<div style='width: 100%; height: 20px; margin-bottom: 5px;'>"
        f"<progress value='{progress_val}' max='{max_val if max_val > 0 else 1}' "
        f"style='width: 100%; height: 20px; color: {color};'></progress>"
        "</div>"
    )

def build_app_interface(selected_lang: str = "zh"):
    """
    Top-level UI function
    Implemented:
        · Logic for new model/model name, automatic directory, dropdown refresh/delete
        · Language switching: After switching the `lang_select` dropdown, **all component labels & default values** are refreshed synchronously
        · Dynamic HTML/SVG loss plot
    """

    T = LANG_JSON[selected_lang]

    # Helper function to match device values
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
        if config_device == 'cuda':
            for device in available_devices:
                if device.startswith('cuda:'):
                    return device
        
        # If config has 'cuda:X' but that specific device is not available, 
        # fall back to first available CUDA device
        if config_device.startswith('cuda:'):
            for device in available_devices:
                if device.startswith('cuda:'):
                    return device
        
        # Default fallback to CPU if available, or first device in list
        if 'cpu' in available_devices:
            return 'cpu'
        
        return available_devices[0] if available_devices else 'cpu'

    # --- UI Helper functions ---
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

    custom_css = """
    .gradio-container{font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Oxygen,Ubuntu,Cantarell,'Open Sans','Helvetica Neue',sans-serif;}
    progress{width:100%;height:20px;margin:4px 0;}
    #train-log-box{
        height:300px;
        overflow-y:auto !important;
        display:block !important;
        font-family:monospace;
        padding:8px;
        background:white;
        border:1px solid #ddd;
        white-space:pre-wrap;
        word-break:break-word;
    }
    """

    with gr.Blocks(title=T["app_title"], css=custom_css) as demo:

        # ========= Top: model management / language ========= #
        with gr.Row():
            model_dropdown = gr.Dropdown(label=T["registered_models"], choices=_get_model_choices_list(), value=None, interactive=True)
            refresh_models_btn = gr.Button(T["refresh_tables"])
            delete_model_btn = gr.Button(T["delete_selected_model"], variant="stop")

        lang_select = gr.Dropdown(label=T["language_label"],
                                    choices=list(LANG_JSON.keys()),
                                    value=selected_lang, interactive=True)

        # ========= Tabs ========= #
        with gr.Tabs() as main_tabs:

            # -------------- Data processing Tab -------------- #
            with gr.Tab(T["data_process_tab"]) as data_process_tab:
                with gr.Row():
                    input_text = gr.Textbox(label=T["dp_paste_text"], lines=19.5)
                    with gr.Column():
                        txt_dir = gr.Textbox(label=T["dp_txt_dir"], value="")
                        new_model_chk = gr.Checkbox(label=T["new_model"], value=True)
                        model_name_box = gr.Textbox(label=T["model_name"], value="new_model")
                        with gr.Row():
                            no_val_set = gr.Checkbox(label=T["dp_no_val_set"],
                                                     value=DEFAULT_CONFIG["data_process"]["no_validation"])
                            use_gpt2 = gr.Checkbox(label=T["dp_use_gpt2_tokenizer"],
                                                     value=DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"])
                        train_split = gr.Slider(label=T["dp_train_split"],
                                                minimum=0.1, maximum=0.99, step=0.01,
                                                value=DEFAULT_CONFIG["data_process"]["train_split_ratio"])
                        num_proc = gr.Number(label=T["dp_num_proc"],
                                             value=DEFAULT_CONFIG["data_process"]["num_proc"],
                                             precision=0)
                process_btn = gr.Button(T["dp_start_btn"])
                process_output = gr.Textbox(label=T["dp_result"], lines=5, interactive=False)

            # -------------- Training Tab -------------- #
            with gr.Tab(T["train_tab"]) as train_tab:
                train_params_title_md = gr.Markdown(f"### {T['train_params_title']}")

                with gr.Row():
                    data_dir_box = gr.Textbox(label=T["train_data_dir"], value="", interactive=False)
                    out_dir_box = gr.Textbox(label=T["train_out_dir"], value="", interactive=False)
                    backend_box = gr.Dropdown(label=T["train_backend"], choices=["nccl", "gloo"], value=DEFAULT_CONFIG["training"]["backend"])
                    available_devices = device_manager.get_available_devices_list()
                    device_box = gr.Dropdown(label=T["train_device"], choices=available_devices,
                                             value=_match_device_value(DEFAULT_CONFIG["training"]["device"], available_devices))
                    dtype_box = gr.Dropdown(label=T["train_dtype"], choices=["float16", "bfloat16", "float32"],
                                            value=DEFAULT_CONFIG["training"]["dtype"])
                    compile_box = gr.Checkbox(label=T["train_compile_model"],
                                              value=DEFAULT_CONFIG["training"]["compile_model"])

                with gr.Row():
                    plot_interval_box = gr.Number(label=T["train_eval_interval"],
                                                  value=DEFAULT_CONFIG["training"]["plot_interval"])
                    log_interval_box = gr.Number(label=T["train_log_interval"],
                                                 value=DEFAULT_CONFIG["training"]["log_interval"])
                    num_eval_seeds_box = gr.Number(label=T["train_num_eval_seeds"],
                                                   value=DEFAULT_CONFIG["training"]["num_eval_seeds"])
                    save_best_val_ckpt_box = gr.Checkbox(label=T["train_save_best_val_ckpt"],
                                                         value=DEFAULT_CONFIG["training"]["save_best_val_checkpoint"])
                    init_from_box = gr.Dropdown(label=T["train_init_from"],
                                                choices=["scratch", "resume"],
                                                value=DEFAULT_CONFIG["training"]["init_from"])
                    seed_box = gr.Number(label=T["train_seed"],
                                         value=DEFAULT_CONFIG["training"]["seed"])

                with gr.Row():
                    grad_acc_box = gr.Number(label=T["train_gas"],
                                             value=DEFAULT_CONFIG["training"]["gradient_accumulation_steps"])
                    batch_size_box = gr.Number(label=T["train_batch_size"],
                                               value=DEFAULT_CONFIG["training"]["batch_size"])
                    block_size_box = gr.Number(label=T["train_block_size"],
                                               value=DEFAULT_CONFIG["training"]["block_size"])
                    n_layer_box = gr.Number(label=T["train_n_layer"],
                                            value=DEFAULT_CONFIG["training"]["n_layer"])
                    n_head_box = gr.Number(label=T["train_n_head"],
                                           value=DEFAULT_CONFIG["training"]["n_head"])
                    n_embd_box = gr.Number(label=T["train_n_embd"],
                                           value=DEFAULT_CONFIG["training"]["n_embd"])

                with gr.Row():
                    dropout_box = gr.Number(label=T["train_dropout"],
                                            value=DEFAULT_CONFIG["training"]["dropout"])
                    bias_box = gr.Checkbox(label=T["train_bias"],
                                           value=DEFAULT_CONFIG["training"]["bias"])
                    lr_box = gr.Number(label=T["train_lr"],
                                       value=DEFAULT_CONFIG["training"]["learning_rate"])
                    max_iters_box = gr.Number(label=T["train_max_iters"],
                                              value=DEFAULT_CONFIG["training"]["max_iters"])
                    weight_decay_box = gr.Number(label=T["train_weight_decay"],
                                                 value=DEFAULT_CONFIG["training"]["weight_decay"])

                with gr.Row():
                    beta1_box = gr.Number(label=T["train_beta1"],
                                          value=DEFAULT_CONFIG["training"]["beta1"])
                    beta2_box = gr.Number(label=T["train_beta2"],
                                          value=DEFAULT_CONFIG["training"]["beta2"])
                    lr_scheduler_box = gr.Dropdown(label=T["train_lr_scheduler"],
                                                   choices=["none", "cosine", "constant_with_warmup",
                                                            "linear", "step", "polynomial"],
                                                   value=DEFAULT_CONFIG["training"]["lr_scheduler_type"])
                    warmup_box = gr.Number(label=T["train_warmup_iters"],
                                           value=DEFAULT_CONFIG["training"]["warmup_iters"])
                    lr_decay_box = gr.Number(label=T["train_lr_decay_iters"],
                                             value=DEFAULT_CONFIG["training"]["lr_decay_iters"])
                    min_lr_box = gr.Number(label=T["train_min_lr"],
                                           value=DEFAULT_CONFIG["training"]["min_lr"])
                    
                with gr.Row():
                    step_size_box = gr.Number(label="Step Size",
                                               value=DEFAULT_CONFIG["training"]["step_size"])
                    step_gamma_box = gr.Number(label="Step Gamma",
                                                value=DEFAULT_CONFIG["training"]["step_gamma"])
                    polynomial_power_box = gr.Number(label="Polynomial Power",
                                                      value=DEFAULT_CONFIG["training"]["polynomial_power"])
                    save_interval_box = gr.Number(label=T["train_save_interval"],
                                                   value=DEFAULT_CONFIG["training"]["save_interval"])

                # Self-attention parameters in collapsible accordion
                with gr.Accordion(label=T["train_self_attn_title"], open=False) as self_attn_accordion:
                    use_self_attention_box = gr.Checkbox(
                        label=T["train_use_self_attention"],
                        value=DEFAULT_CONFIG["training"]["use_self_attention"]
                    )
                    
                    with gr.Row():
                        ffn_hidden_mult_box = gr.Number(
                            label=T["train_ffn_hidden_mult"],
                            value=DEFAULT_CONFIG["training"]["ffn_hidden_mult"],
                            visible=False
                        )
                        qkv_bias_box = gr.Checkbox(
                            label=T["train_qkv_bias"],
                            value=DEFAULT_CONFIG["training"]["qkv_bias"],
                            visible=False
                        )
                        attn_dropout_box = gr.Number(
                            label=T["train_attn_dropout"],
                            value=DEFAULT_CONFIG["training"]["attn_dropout"],
                            step=0.01,
                            visible=False
                        )
                        resid_dropout_box = gr.Number(
                            label=T["train_resid_dropout"],
                            value=DEFAULT_CONFIG["training"]["resid_dropout"],
                            step=0.01,
                            visible=False
                        )
                    
                    with gr.Row():
                        ln_eps_box = gr.Number(
                            label=T["train_ln_eps"],
                            value=DEFAULT_CONFIG["training"]["ln_eps"],
                            step=1e-6,
                            visible=False
                        )
                        init_std_box = gr.Number(
                            label=T["train_init_std"],
                            value=DEFAULT_CONFIG["training"]["init_std"],
                            step=0.001,
                            visible=False
                        )
                        use_flash_attn_box = gr.Checkbox(
                            label=T["train_use_flash_attn"],
                            value=DEFAULT_CONFIG["training"]["use_flash_attn"],
                            visible=False
                        )
                        pos_encoding_type_box = gr.Dropdown(
                            label=T["train_pos_encoding_type"],
                            choices=["rope", "alibi"],
                            value=DEFAULT_CONFIG["training"]["pos_encoding_type"],
                            visible=False
                        )
                    
                    # New optimized parameters - row 1
                    with gr.Row():
                        rope_base_box = gr.Number(
                            label=T["train_rope_base"],
                            value=DEFAULT_CONFIG["training"]["rope_base"],
                            visible=False
                        )
                        rope_cache_size_box = gr.Number(
                            label=T["train_rope_cache_size"],
                            value=DEFAULT_CONFIG["training"]["rope_cache_size"],
                            visible=False,
                            info="Cache size for RoPE (0 for auto)"
                        )
                        alibi_bias_scale_box = gr.Number(
                            label=T["train_alibi_bias_scale"],
                            value=DEFAULT_CONFIG["training"]["alibi_bias_scale"],
                            step=0.1,
                            visible=False,
                            info="Scaling factor for ALiBi bias"
                        )
                        ffn_activation_box = gr.Dropdown(
                            label=T["train_ffn_activation"],
                            choices=["gelu", "relu", "swish"],
                            value=DEFAULT_CONFIG["training"]["ffn_activation"],
                            visible=False,
                            info="FFN activation function"
                        )
                    
                    # New optimized parameters - row 2
                    with gr.Row():
                        attention_scale_factor_box = gr.Number(
                            label=T["train_attention_scale_factor"],
                            value=DEFAULT_CONFIG["training"]["attention_scale_factor"],
                            step=0.1,
                            visible=False,
                            info="Additional attention scaling"
                        )
                        gradient_checkpointing_box = gr.Checkbox(
                            label=T["train_gradient_checkpointing"],
                            value=DEFAULT_CONFIG["training"]["gradient_checkpointing"],
                            visible=False,
                            info="Enable gradient checkpointing to save memory"
                        )
                        cache_strategy_box = gr.Dropdown(
                            label=T["train_cache_strategy"],
                            choices=["adaptive", "fixed", "minimal"],
                            value=DEFAULT_CONFIG["training"]["cache_strategy"],
                            visible=False,
                            info="Cache allocation strategy"
                        )
                        max_cache_size_box = gr.Number(
                            label=T["train_max_cache_size"],
                            value=DEFAULT_CONFIG["training"]["max_cache_size"],
                            visible=False,
                            info="Maximum cache size for dynamic allocation"
                        )
                    
                    # Error handling parameters
                    with gr.Row():
                        strict_validation_box = gr.Checkbox(
                            label=T["train_strict_validation"],
                            value=DEFAULT_CONFIG["training"]["strict_validation"],
                            visible=False,
                            info="Enable strict input validation"
                        )
                        fallback_on_error_box = gr.Checkbox(
                            label=T["train_fallback_on_error"],
                            value=DEFAULT_CONFIG["training"]["fallback_on_error"],
                            visible=False,
                            info="Fallback to basic implementations on error"
                        )

                train_btn = gr.Button(T["train_start_btn"])
                stop_btn = gr.Button(T["stop_btn"])

                with gr.Row():
                    with gr.Column(scale=1):
                        train_progress = gr.HTML(label="Training Progress")
                        train_log = gr.HTML(label=T["train_log"], elem_id="train-log-box")
                    with gr.Column(scale=2):
                        train_plot = gr.HTML(label=T["train_plot"]) # Changed from gr.Image

            # -------------- SFT Tab -------------- #
            with gr.Tab(T["sft_tab"]) as sft_tab:
                gr.Markdown("### Supervised Fine-Tuning (SFT)")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sft_base_model = gr.Dropdown(
                            label=T["sft_base_model"],
                            choices=_get_model_choices_list(),
                            value=None,
                            interactive=True
                        )
                        sft_refresh_model_btn = gr.Button("🔄 Refresh Models", size="sm")
                    
                    with gr.Column(scale=2):
                        sft_dataset_file = gr.File(
                            label=T["sft_dataset_file"],
                            file_types=[".json"],
                            type="filepath"
                        )
                        sft_dataset_dir = gr.Textbox(
                            label=T["sft_dataset_dir"],
                            placeholder="Or enter directory path containing JSON files..."
                        )
                
                sft_format_status = gr.Textbox(
                    label=T["sft_format_status"],
                    value=T["sft_no_dataset"],
                    interactive=False
                )
                sft_validate_btn = gr.Button("🔍 Validate Dataset")
                
                # Store loaded dataset in state
                sft_dataset_state = gr.State(value=[])
                
                with gr.Row():
                    with gr.Column():
                        sft_epochs = gr.Number(
                            label=T["sft_epochs"],
                            value=DEFAULT_CONFIG["sft"]["epochs"],
                            minimum=1, maximum=100
                        )
                        sft_batch_size = gr.Number(
                            label=T["sft_batch_size"],
                            value=DEFAULT_CONFIG["sft"]["batch_size"],
                            minimum=1, maximum=64
                        )
                        sft_gradient_accumulation = gr.Number(
                            label=T["sft_gradient_accumulation"],
                            value=DEFAULT_CONFIG["sft"]["gradient_accumulation_steps"],
                            minimum=1, maximum=32
                        )
                    with gr.Column():
                        sft_learning_rate = gr.Number(
                            label=T["sft_learning_rate"],
                            value=DEFAULT_CONFIG["sft"]["learning_rate"],
                            step=1e-6
                        )
                        sft_max_seq_length = gr.Number(
                            label=T["sft_max_seq_length"],
                            value=DEFAULT_CONFIG["sft"]["max_seq_length"],
                            minimum=32, maximum=4096
                        )
                        sft_warmup_ratio = gr.Number(
                            label=T["sft_warmup_ratio"],
                            value=DEFAULT_CONFIG["sft"]["warmup_ratio"],
                            step=0.01, minimum=0, maximum=1
                        )
                
                sft_system_prompt = gr.Textbox(
                    label=T["sft_system_prompt"],
                    value=DEFAULT_CONFIG["sft"]["system_prompt"],
                    lines=2
                )
                
                with gr.Row():
                    sft_start_btn = gr.Button(T["sft_start_btn"], variant="primary")
                    sft_stop_btn = gr.Button(T["sft_stop_btn"], variant="stop")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        sft_progress = gr.HTML(label="SFT Progress")
                        sft_log = gr.HTML(label=T["sft_log"], elem_id="sft-log-box")
                    with gr.Column(scale=2):
                        sft_plot = gr.HTML(label=T["sft_plot"])

            # -------------- Inference Tab -------------- #
            with gr.Tab(T["infer_tab"]) as inf_tab:
                with gr.Row():
                    data_dir_inf = gr.Textbox(label=T["dp_processed_dir"], value="", interactive=False)
                    out_dir_inf = gr.Textbox(label=T["inf_out_dir"], value="", interactive=False)

                prompt_box = gr.Textbox(label=T["inf_prompt"],
                                          value=DEFAULT_CONFIG["inference"]["prompt"], lines=5,
                                          placeholder="Just write something...")

                with gr.Row():
                    num_samples_box = gr.Number(label=T["inf_num_samples"],
                                                value=DEFAULT_CONFIG["inference"]["num_samples"])
                    max_new_tokens_box = gr.Number(label=T["inf_max_new_tokens"],
                                                   value=DEFAULT_CONFIG["inference"]["max_new_tokens"])
                    temperature_box = gr.Number(label=T["inf_temperature"],
                                                value=DEFAULT_CONFIG["inference"]["temperature"],
                                                step=0.1)
                    top_k_box = gr.Number(label=T["inf_top_k"],
                                          value=DEFAULT_CONFIG["inference"]["top_k"])
                    dtype_box_inf = gr.Dropdown(label=T["inf_dtype"],
                                               choices=["float16", "bfloat16", "float32"],
                                               value=DEFAULT_CONFIG["inference"]["dtype"])
                    available_devices_inf = device_manager.get_available_devices_list()
                    device_box_inf = gr.Dropdown(label=T["inf_device"],
                                                choices=available_devices_inf,
                                                value=_match_device_value(DEFAULT_CONFIG["inference"]["device"], available_devices_inf))
                    seed_box_inf = gr.Number(label=T["inf_seed"],
                                             value=DEFAULT_CONFIG["inference"]["seed"])

                # Chat Mode Toggle
                inf_chat_mode = gr.Checkbox(label=T["inf_chat_mode"], value=False)
                
                # Chat Interface (Hidden by default, shown when Chat Mode is enabled)
                with gr.Group(visible=False) as chat_interface_group:
                    chatbot = gr.Chatbot(
                        label=T["inf_chat_history"], 
                        height=400,
                        sanitize_html=False  # Allow HTML rendering for token highlighting
                    )
                    inf_system_prompt = gr.Textbox(
                        label=T["inf_system_prompt"], 
                        value=DEFAULT_CONFIG["sft"]["system_prompt"]
                    )
                    with gr.Row():
                        inf_user_input = gr.Textbox(
                            label=T["inf_user_input"], 
                            placeholder="Type a message...",
                            scale=4
                        )
                        with gr.Column(scale=1, min_width=100):
                            inf_send_btn = gr.Button(T["inf_send_btn"], variant="primary")
                            inf_clear_btn = gr.Button(T["inf_clear_chat"])
                    
                    # Chat advanced output section (collapsed by default)
                    with gr.Accordion(T["inf_chat_advanced"], open=False) as chat_advanced_accordion:
                        # Response token details table only (user tokenization is now shown inline in message bubble)
                        chat_advanced_output = gr.HTML(label=T["inf_advanced_output"], elem_id="chat-advanced-html")
                
                # Standard Inference Interface (Hidden when Chat Mode is enabled)
                with gr.Group(visible=True) as standard_interface_group:
                    inf_btn = gr.Button(T["inf_start_btn"])
                    # Use HTML for token-highlighted output
                    inf_output = gr.HTML(label=T["inf_result"], elem_id="inf-result-html")
                    
                    # Advanced output section (collapsed by default)
                    with gr.Accordion(T["inf_advanced_output"], open=False) as advanced_accordion:
                        inf_advanced_output = gr.HTML(label=T["inf_advanced_output"], elem_id="inf-advanced-html")

                # Visibility Logic
                def toggle_chat_mode(is_chat):
                    return {
                        chat_interface_group: gr.update(visible=is_chat),
                        standard_interface_group: gr.update(visible=not is_chat),
                        prompt_box: gr.update(visible=not is_chat)
                    }
                
                inf_chat_mode.change(
                    fn=toggle_chat_mode,
                    inputs=[inf_chat_mode],
                    outputs=[chat_interface_group, standard_interface_group, prompt_box]
                )

            # -------------- Comparison Tab -------------- #
            with gr.Tab(T["compare_tab"]) as comp_tab:
                # Two-column layout for model comparison
                with gr.Row():
                    with gr.Column():
                        comp_left_model = gr.Dropdown(label=T["compare_left_model"], choices=_get_model_choices_list(), value=None, interactive=True)
                    with gr.Column():
                        comp_right_model = gr.Dropdown(label=T["compare_right_model"], choices=_get_model_choices_list(), value=None, interactive=True)
                
                # Model parameters display
                with gr.Row():
                    with gr.Column():
                        comp_left_params = gr.JSON(label=T["compare_model_params"], value={})
                    with gr.Column():
                        comp_right_params = gr.JSON(label=T["compare_model_params"], value={})
                
                # Loss curves
                with gr.Row():
                    with gr.Column():
                        comp_left_plot = gr.HTML(label=T["compare_loss_curve"])
                    with gr.Column():
                        comp_right_plot = gr.HTML(label=T["compare_loss_curve"])
                
                # Inference history
                with gr.Row():
                    with gr.Column():
                        comp_left_history = gr.Textbox(label=T["compare_inference_history"], lines=5)
                    with gr.Column():
                        comp_right_history = gr.Textbox(label=T["compare_inference_history"], lines=5)
                
                # Inference playground
                gr.Markdown(f"### {T['compare_inference_playground']}")
                
                # 为左右两个模型分别设置参数
                with gr.Row():
                    # 左侧模型参数
                    with gr.Column():
                        gr.Markdown(f"**⚙️ Model 1**")
                        with gr.Row():
                            comp_left_num_samples = gr.Number(label=T["inf_num_samples"], 
                                                           value=DEFAULT_CONFIG["inference"]["num_samples"])
                            comp_left_max_tokens = gr.Number(label=T["inf_max_new_tokens"], 
                                                           value=DEFAULT_CONFIG["inference"]["max_new_tokens"])
                            comp_left_temperature = gr.Number(label=T["inf_temperature"], 
                                                            value=DEFAULT_CONFIG["inference"]["temperature"],
                                                            step=0.1)
                        with gr.Row():
                            comp_left_top_k = gr.Number(label=T["inf_top_k"], 
                                                      value=DEFAULT_CONFIG["inference"]["top_k"])
                            comp_left_dtype = gr.Dropdown(label=T["inf_dtype"],
                                                        choices=["float16", "bfloat16", "float32"],
                                                        value=DEFAULT_CONFIG["inference"]["dtype"])
                            comp_left_seed = gr.Number(label=T["inf_seed"], 
                                                     value=DEFAULT_CONFIG["inference"]["seed"])
                
                    # 右侧模型参数
                    with gr.Column():
                        gr.Markdown(f"**⚙️ Model 2**")
                        with gr.Row():
                            comp_right_num_samples = gr.Number(label=T["inf_num_samples"], 
                                                             value=DEFAULT_CONFIG["inference"]["num_samples"])
                            comp_right_max_tokens = gr.Number(label=T["inf_max_new_tokens"], 
                                                             value=DEFAULT_CONFIG["inference"]["max_new_tokens"])
                            comp_right_temperature = gr.Number(label=T["inf_temperature"], 
                                                             value=DEFAULT_CONFIG["inference"]["temperature"],
                                                             step=0.1)
                        with gr.Row():
                            comp_right_top_k = gr.Number(label=T["inf_top_k"], 
                                                       value=DEFAULT_CONFIG["inference"]["top_k"])
                            comp_right_dtype = gr.Dropdown(label=T["inf_dtype"],
                                                         choices=["float16", "bfloat16", "float32"],
                                                         value=DEFAULT_CONFIG["inference"]["dtype"])
                            comp_right_seed = gr.Number(label=T["inf_seed"], 
                                                    value=DEFAULT_CONFIG["inference"]["seed"])

                # Shared prompt
                comp_prompt = gr.Textbox(label=T["compare_shared_prompt"], lines=5, 
                                         value=DEFAULT_CONFIG["inference"]["prompt"],
                                         placeholder="Just write something...")
                
                # Generate button
                comp_generate_btn = gr.Button(T["compare_generate_btn"])
                
                # Output display
                with gr.Row():
                    with gr.Column():
                        comp_left_output = gr.Textbox(label=T["compare_left_output"], lines=10)
                    with gr.Column():
                        comp_right_output = gr.Textbox(label=T["compare_right_output"], lines=10)
                
                # Hidden fields to store model data paths
                comp_left_data_dir = gr.Textbox(visible=False)
                comp_left_out_dir = gr.Textbox(visible=False)
                comp_right_data_dir = gr.Textbox(visible=False)
                comp_right_out_dir = gr.Textbox(visible=False)

        # ------------------------------------------------------------------
        # Call backs: data processing / training / inference
        def data_processing_cb(
            new_flag, model_name, dropdown_val,
            txt, ddir,
            sp, no_val, use_gpt2_tokenizer, num_proc_
        ):
            try:
                current_lang = lang_select.value # Direct access to component's value
                T_current = LANG_JSON[current_lang]
                
                sel_id = int(dropdown_val.split(" - ")[0]) if dropdown_val and " - " in dropdown_val else None
                info = process_data(
                    model_name=model_name.strip() or "unnamed",
                    new_model=new_flag,
                    selected_model_id=sel_id,
                    input_text=txt,
                    input_dir=ddir,
                    train_split_ratio=sp,
                    no_validation=no_val,
                    use_gpt2_tokenizer=use_gpt2_tokenizer,
                    num_proc=int(num_proc_) # num_proc_ is from gr.Number, so float then int
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
                # 同时更新主下拉框和对比页面的两个下拉框
                return msg, gr.update(choices=new_choices, value=new_val), gr.update(choices=new_choices), gr.update(choices=new_choices)
            except Exception as e:
                return f"❌ Error: {str(e)}", gr.update(), gr.update(), gr.update()

        process_btn.click(
            fn=data_processing_cb,
            inputs=[new_model_chk, model_name_box, model_dropdown,
                    input_text, txt_dir,
                    train_split, no_val_set, use_gpt2, num_proc],
            outputs=[process_output, model_dropdown, comp_left_model, comp_right_model]
        )

        stop_btn.click(fn=stop_training, inputs=[], outputs=[])
        
        # -----------------------------
        # LR Scheduler Callback
        # -----------------------------
        def update_lr_scheduler_params(scheduler_type):
            defaults_train = DEFAULT_CONFIG["training"]
            
            # 初始化所有组件为非交互状态，视觉上显示为空白（但实际值保留为默认值）
            # 注意：这里使用空字符串作为显示值，但实际值会保留在后端，懒得改接口了。
            warmup_update = gr.update(interactive=False, value="") 
            lr_decay_update = gr.update(interactive=False, value="")
            min_lr_update = gr.update(interactive=False, value="")
            step_size_update = gr.update(interactive=False, value="")
            step_gamma_update = gr.update(interactive=False, value="")
            polynomial_power_update = gr.update(interactive=False, value="")
            
            if scheduler_type == "none":
                pass # 所有参数保持非交互状态
            elif scheduler_type == "cosine":
                # 启用相关参数并设置其值
                warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
                lr_decay_update = gr.update(interactive=True, value=defaults_train["lr_decay_iters"])
                min_lr_update = gr.update(interactive=True, value=defaults_train["min_lr"])
            elif scheduler_type == "constant_with_warmup":
                warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
            elif scheduler_type == "linear":
                warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
                lr_decay_update = gr.update(interactive=True, value=defaults_train["lr_decay_iters"])
                min_lr_update = gr.update(interactive=True, value=defaults_train["min_lr"])
            elif scheduler_type == "step":
                step_size_update = gr.update(interactive=True, value=defaults_train["step_size"])
                step_gamma_update = gr.update(interactive=True, value=defaults_train["step_gamma"])
            elif scheduler_type == "polynomial":
                warmup_update = gr.update(interactive=True, value=defaults_train["warmup_iters"])
                lr_decay_update = gr.update(interactive=True, value=defaults_train["lr_decay_iters"])
                min_lr_update = gr.update(interactive=True, value=defaults_train["min_lr"])
                polynomial_power_update = gr.update(interactive=True, value=defaults_train["polynomial_power"])
                
            return [
                warmup_update, lr_decay_update, min_lr_update,
                step_size_update, step_gamma_update, polynomial_power_update
            ]

        lr_scheduler_box.change(
            fn=update_lr_scheduler_params,
            inputs=[lr_scheduler_box],
            outputs=[
                warmup_box, lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box
            ]
        )
        
        # -----------------------------
        # Self-Attention Parameters Callback
        # -----------------------------
        def update_self_attention_params(use_self_attention):
            """Update visibility of self-attention parameters based on checkbox"""
            defaults_train = DEFAULT_CONFIG["training"]
            
            if use_self_attention:
                return [
                    gr.update(visible=True, value=defaults_train["ffn_hidden_mult"]),  # ffn_hidden_mult_box
                    gr.update(visible=True, value=defaults_train["qkv_bias"]),         # qkv_bias_box
                    gr.update(visible=True, value=defaults_train["attn_dropout"]),     # attn_dropout_box
                    gr.update(visible=True, value=defaults_train["resid_dropout"]),    # resid_dropout_box
                    gr.update(visible=True, value=defaults_train["ln_eps"]),           # ln_eps_box
                    gr.update(visible=True, value=defaults_train["init_std"]),         # init_std_box
                    gr.update(visible=True, value=defaults_train["use_flash_attn"]),   # use_flash_attn_box
                    gr.update(visible=True, value=defaults_train["pos_encoding_type"]), # pos_encoding_type_box
                    gr.update(visible=True, value=defaults_train["rope_base"]),        # rope_base_box
                    # New optimized parameters
                    gr.update(visible=True, value=defaults_train["rope_cache_size"]),  # rope_cache_size_box
                    gr.update(visible=True, value=defaults_train["alibi_bias_scale"]), # alibi_bias_scale_box
                    gr.update(visible=True, value=defaults_train["ffn_activation"]),   # ffn_activation_box
                    gr.update(visible=True, value=defaults_train["attention_scale_factor"]), # attention_scale_factor_box
                    gr.update(visible=True, value=defaults_train["gradient_checkpointing"]), # gradient_checkpointing_box
                    gr.update(visible=True, value=defaults_train["cache_strategy"]),   # cache_strategy_box
                    gr.update(visible=True, value=defaults_train["max_cache_size"]),   # max_cache_size_box
                    gr.update(visible=True, value=defaults_train["strict_validation"]), # strict_validation_box
                    gr.update(visible=True, value=defaults_train["fallback_on_error"]) # fallback_on_error_box
                ]
            else:
                return [
                    gr.update(visible=False),  # ffn_hidden_mult_box
                    gr.update(visible=False),  # qkv_bias_box
                    gr.update(visible=False),  # attn_dropout_box
                    gr.update(visible=False),  # resid_dropout_box
                    gr.update(visible=False),  # ln_eps_box
                    gr.update(visible=False),  # init_std_box
                    gr.update(visible=False),  # use_flash_attn_box
                    gr.update(visible=False),  # pos_encoding_type_box
                    gr.update(visible=False),  # rope_base_box
                    # New optimized parameters
                    gr.update(visible=False),  # rope_cache_size_box
                    gr.update(visible=False),  # alibi_bias_scale_box
                    gr.update(visible=False),  # ffn_activation_box
                    gr.update(visible=False),  # attention_scale_factor_box
                    gr.update(visible=False),  # gradient_checkpointing_box
                    gr.update(visible=False),  # cache_strategy_box
                    gr.update(visible=False),  # max_cache_size_box
                    gr.update(visible=False),  # strict_validation_box
                    gr.update(visible=False)   # fallback_on_error_box
                ]

        use_self_attention_box.change(
            fn=update_self_attention_params,
            inputs=[use_self_attention_box],
            outputs=[
                ffn_hidden_mult_box, qkv_bias_box, attn_dropout_box, resid_dropout_box,
                ln_eps_box, init_std_box, use_flash_attn_box, pos_encoding_type_box,
                rope_base_box,
                # New optimized parameters
                rope_cache_size_box, alibi_bias_scale_box, ffn_activation_box, attention_scale_factor_box,
                gradient_checkpointing_box, cache_strategy_box, max_cache_size_box,
                strict_validation_box, fallback_on_error_box
            ]
        )
        
        # ------------------------------------------------------------------ #
        # Call backs: start training
        # ------------------------------------------------------------------ #
        def training_cb(
            data_dir_, out_dir_, plot_interval_, log_interval_, num_eval_seeds_,
            save_best_val_ckpt_, init_from_,
            grad_acc_, batch_size_, block_size_,
            n_layer_, n_head_, n_embd_,
            dropout_, bias_,
            lr_, max_iters_, weight_decay_,
            beta1_, beta2_,
            lr_scheduler_type_, warmup_,
            lr_decay_, min_lr_,
            step_size_, step_gamma_, polynomial_power_,
            backend_, device_, dtype_, compile_,
            seed_, save_interval_,
            # Self-attention parameters
            use_self_attention_, ffn_hidden_mult_, qkv_bias_, attn_dropout_, 
            resid_dropout_, ln_eps_, init_std_, use_flash_attn_, pos_encoding_type_, rope_base_,
            # New optimized parameters
            rope_cache_size_, alibi_bias_scale_, ffn_activation_, attention_scale_factor_,
            gradient_checkpointing_, cache_strategy_, max_cache_size_, strict_validation_, fallback_on_error_
        ):
            # Pre-training cleanup: Clear any residual GPU resources from previous failed training
            try:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                # Also clear model cache to ensure fresh start
                cache = ModelCache()
                cache.clear_cache()
                print("🧹 Pre-training cleanup completed")
            except Exception as cleanup_err:
                print(f"Warning: Pre-training cleanup encountered an issue: {cleanup_err}")
            
            empty_plot_html = generate_loss_chart_html([], [])
            
            # Enhanced input validation with better error messages
            try:
                # Convert Number inputs safely with improved error handling
                num_eval_seeds_int = int(float(num_eval_seeds_))
                if not (0 <= num_eval_seeds_int <= 2**32 - 1):
                    raise ValueError(f"num_eval_seeds must be between 0 and {2**32 - 1}, got {num_eval_seeds_int}")
                
                # Validate new parameters
                rope_cache_size_int = None if rope_cache_size_ == 0 else int(float(rope_cache_size_))
                if rope_cache_size_int is not None and rope_cache_size_int < 0:
                    raise ValueError(f"rope_cache_size must be non-negative or 0 for auto, got {rope_cache_size_int}")
                
                alibi_bias_scale_float = float(alibi_bias_scale_)
                if alibi_bias_scale_float <= 0:
                    raise ValueError(f"alibi_bias_scale must be positive, got {alibi_bias_scale_float}")
                
                attention_scale_factor_float = float(attention_scale_factor_)
                if attention_scale_factor_float <= 0:
                    raise ValueError(f"attention_scale_factor must be positive, got {attention_scale_factor_float}")
                
                max_cache_size_int = int(float(max_cache_size_))
                if max_cache_size_int <= 0:
                    raise ValueError(f"max_cache_size must be positive, got {max_cache_size_int}")
                
                # Validate FFN activation
                if ffn_activation_ not in ["gelu", "relu", "swish"]:
                    raise ValueError(f"ffn_activation must be one of ['gelu', 'relu', 'swish'], got {ffn_activation_}")
                
                # Validate cache strategy
                if cache_strategy_ not in ["adaptive", "fixed", "minimal"]:
                    raise ValueError(f"cache_strategy must be one of ['adaptive', 'fixed', 'minimal'], got {cache_strategy_}")
                    
            except ValueError as e:
                error_msg = f"Parameter validation error: {str(e)}"
                yield (f"<div style='color:red;'>{error_msg}</div>", "", empty_plot_html)
                return

            try:
                defaults_train = DEFAULT_CONFIG["training"]
                def safe_int(v, default_val): 
                    try:
                        return default_val if v == "" or v is None else int(float(v))
                    except (ValueError, TypeError):
                        return default_val
                        
                def safe_float(v, default_val): 
                    try:
                        return default_val if v == "" or v is None else float(v)
                    except (ValueError, TypeError):
                        return default_val
                        
                def safe_bool(v, default_val): 
                    try:
                        return default_val if v is None else bool(v)
                    except (ValueError, TypeError):
                        return default_val
                
                def safe_str(v, default_val):
                    try:
                        return default_val if v is None or v == "" else str(v)
                    except (ValueError, TypeError):
                        return default_val

                gen = train_model_generator(
                    data_dir=data_dir_,
                    out_dir=out_dir_,
                    plot_interval=safe_int(plot_interval_, defaults_train["plot_interval"]),
                    log_interval=safe_int(log_interval_, defaults_train["log_interval"]),
                    num_eval_seeds=num_eval_seeds_int,
                    save_best_val_checkpoint=bool(save_best_val_ckpt_),
                    init_from=init_from_,
                    gradient_accumulation_steps=safe_int(grad_acc_, defaults_train["gradient_accumulation_steps"]),
                    batch_size=safe_int(batch_size_, defaults_train["batch_size"]),
                    block_size=safe_int(block_size_, defaults_train["block_size"]),
                    n_layer=safe_int(n_layer_, defaults_train["n_layer"]),
                    n_head=safe_int(n_head_, defaults_train["n_head"]),
                    n_embd=safe_int(n_embd_, defaults_train["n_embd"]),
                    dropout=safe_float(dropout_, defaults_train["dropout"]),
                    bias=bool(bias_),
                    learning_rate=safe_float(lr_, defaults_train["learning_rate"]),
                    max_iters=safe_int(max_iters_, defaults_train["max_iters"]),
                    weight_decay=safe_float(weight_decay_, defaults_train["weight_decay"]),
                    beta1=safe_float(beta1_, defaults_train["beta1"]),
                    beta2=safe_float(beta2_, defaults_train["beta2"]),
                    lr_scheduler_type=lr_scheduler_type_,
                    warmup_iters=safe_int(warmup_, defaults_train["warmup_iters"]),
                    lr_decay_iters=safe_int(lr_decay_, defaults_train["lr_decay_iters"]),
                    min_lr=safe_float(min_lr_, defaults_train["min_lr"]),
                    step_size=safe_int(step_size_, defaults_train["step_size"]),
                    step_gamma=safe_float(step_gamma_, defaults_train["step_gamma"]),
                    polynomial_power=safe_float(polynomial_power_, defaults_train["polynomial_power"]),
                    backend=backend_, device=device_, dtype=dtype_,
                    compile_model=bool(compile_),
                    seed=safe_int(seed_, defaults_train["seed"]),
                    save_interval=safe_int(save_interval_, defaults_train["save_interval"]),
                    # Self-attention parameters
                    use_self_attention=safe_bool(use_self_attention_, defaults_train["use_self_attention"]),
                    ffn_hidden_mult=safe_int(ffn_hidden_mult_, defaults_train["ffn_hidden_mult"]),
                    qkv_bias=safe_bool(qkv_bias_, defaults_train["qkv_bias"]),
                    attn_dropout=safe_float(attn_dropout_, defaults_train["attn_dropout"]),
                    resid_dropout=safe_float(resid_dropout_, defaults_train["resid_dropout"]),
                    ln_eps=safe_float(ln_eps_, defaults_train["ln_eps"]),
                    init_std=safe_float(init_std_, defaults_train["init_std"]),
                    use_flash_attn=safe_bool(use_flash_attn_, defaults_train["use_flash_attn"]),
                    pos_encoding_type=pos_encoding_type_ if pos_encoding_type_ else defaults_train["pos_encoding_type"],
                    rope_base=safe_int(rope_base_, defaults_train["rope_base"]),
                    # New optimized parameters
                    rope_cache_size=rope_cache_size_int,
                    alibi_bias_scale=alibi_bias_scale_float,
                    ffn_activation=safe_str(ffn_activation_, defaults_train["ffn_activation"]),
                    attention_scale_factor=attention_scale_factor_float,
                    gradient_checkpointing=safe_bool(gradient_checkpointing_, defaults_train["gradient_checkpointing"]),
                    cache_strategy=safe_str(cache_strategy_, defaults_train["cache_strategy"]),
                    max_cache_size=max_cache_size_int,
                    strict_validation=safe_bool(strict_validation_, defaults_train["strict_validation"]),
                    fallback_on_error=safe_bool(fallback_on_error_, defaults_train["fallback_on_error"])
                )
                
                for p_html_progress, log_line_html, plot_data_tuple in gen:
                    current_plot_rendered_html = empty_plot_html # Default if no plot data
                    if plot_data_tuple:
                        tr_steps, tr_losses, val_steps, val_losses = plot_data_tuple
                        train_data_tuples = list(zip(tr_steps, tr_losses)) if tr_steps and tr_losses else []
                        val_data_tuples = list(zip(val_steps, val_losses)) if val_steps and val_losses else []
                        current_plot_rendered_html = generate_loss_chart_html(train_data_tuples, val_data_tuples)
                    
                    yield (p_html_progress, log_line_html, current_plot_rendered_html)
            
            except Exception as e:
                import traceback
                print(f"Training callback error: {traceback.format_exc()}") # Server-side log
                err_msg = f"Runtime Error in Training: {str(e)}"
                
                # Critical: Clean up GPU resources after training failure
                try:
                    import gc
                    # Force garbage collection to release Python objects
                    gc.collect()
                    
                    # Clear CUDA cache if available
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    
                    # Clear model cache to ensure next training starts fresh
                    try:
                        cache = ModelCache()
                        cache.clear_cache()
                    except Exception as cache_err:
                        print(f"Warning: Failed to clear model cache: {cache_err}")
                    
                    print("🧹 Post-error cleanup completed - GPU resources released")
                except Exception as cleanup_err:
                    print(f"Warning: Post-error cleanup failed: {cleanup_err}")
                
                yield (f"<div style='color:red;'>{err_msg}</div>", "", empty_plot_html)

        train_btn.click(
            fn=training_cb,
            inputs=[
                data_dir_box, out_dir_box,
                plot_interval_box, log_interval_box, num_eval_seeds_box,
                save_best_val_ckpt_box, init_from_box,
                grad_acc_box, batch_size_box, block_size_box,
                n_layer_box, n_head_box, n_embd_box,
                dropout_box, bias_box,
                lr_box, max_iters_box, weight_decay_box,
                beta1_box, beta2_box,
                lr_scheduler_box, warmup_box,
                lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box,
                backend_box, device_box, dtype_box, compile_box,
                seed_box, save_interval_box,
                # Self-attention parameters
                use_self_attention_box, ffn_hidden_mult_box, qkv_bias_box, attn_dropout_box,
                resid_dropout_box, ln_eps_box, init_std_box, use_flash_attn_box, pos_encoding_type_box,
                rope_base_box,
                # New optimized parameters
                rope_cache_size_box, alibi_bias_scale_box, ffn_activation_box, attention_scale_factor_box,
                gradient_checkpointing_box, cache_strategy_box, max_cache_size_box,
                strict_validation_box, fallback_on_error_box
            ],
            outputs=[train_progress, train_log, train_plot]
        )

        # ------------------------------------------------------------------
        # Call backs: inference
        
        # Token highlight colors - alternating colors for adjacent tokens
        TOKEN_COLORS = [
            "#FFE066",  # Yellow
            "#98D8AA",  # Light green
            "#87CEEB",  # Sky blue
            "#DDA0DD",  # Plum
            "#F0B27A",  # Light orange
            "#AED6F1",  # Light blue
            "#F9E79F",  # Light yellow
            "#D5A6BD",  # Light purple
        ]
        
        def _escape_html(text):
            """Escape HTML special characters"""
            return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;").replace("\n", "<br>")
        
        def _generate_token_html(tokens_with_text, prompt_text="", prompt_tokens=None):
            """
            Generate HTML with token highlighting
            tokens_with_text: list of dicts with 'text' and optionally 'token_detail'
            prompt_text: raw prompt text (used if prompt_tokens is None)
            prompt_tokens: list of dicts with 'text', 'original_id', 'mapped_id', 'in_vocab' for each prompt token
            """
            html_parts = []
            html_parts.append('<div style="font-family: monospace; white-space: pre-wrap; line-height: 1.6; padding: 10px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0;">')
            
            # Add prompt part with RED border box to highlight user input
            if prompt_tokens:
                html_parts.append('<span style="display: inline; border: 2px solid #e53935; border-radius: 4px; padding: 2px 4px; background: rgba(229, 57, 53, 0.08); margin-right: 4px;">')
                for i, token_info in enumerate(prompt_tokens):
                    text = token_info.get('text', '')
                    if not text:
                        continue
                    escaped_text = _escape_html(text)
                    color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
                    orig_id = token_info.get('original_id', '?')
                    mapped_id = token_info.get('mapped_id', '?')
                    in_vocab = token_info.get('in_vocab', True)
                    border_color = "#4caf50" if in_vocab else "#f44336"
                    tooltip = f"Prompt Token #{i+1}&#10;Text: '{escaped_text}'&#10;Original ID: {orig_id}&#10;Mapped ID: {mapped_id}&#10;In Vocab: {'Yes' if in_vocab else 'No'}"
                    html_parts.append(f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 3px; border-bottom: 2px solid {border_color}; cursor: help;" title="{tooltip}">{escaped_text}</span>')
                html_parts.append('</span>')
            elif prompt_text:
                # Fallback: RED border box for prompt without token details
                escaped_prompt = _escape_html(prompt_text)
                html_parts.append(f'<span style="display: inline; border: 2px solid #e53935; border-radius: 4px; padding: 2px 4px; background: rgba(229, 57, 53, 0.08); color: #333;">{escaped_prompt}</span>')
            
            # Add generated tokens with highlighting
            color_idx = 0
            for item in tokens_with_text:
                text = item.get('text', '')
                if not text:
                    continue
                
                escaped_text = _escape_html(text)
                color = TOKEN_COLORS[color_idx % len(TOKEN_COLORS)]
                
                # Create highlighted span with tooltip
                token_detail = item.get('token_detail')
                if token_detail:
                    candidates = token_detail.get('top5_candidates', [])
                    selected_id = token_detail.get('selected_token_id')
                    tooltip_parts = [f"#{token_detail.get('position', '?')}: {escaped_text}"]
                    for i, cand in enumerate(candidates[:6]):  # Show up to 6 (selected + top 5)
                        prob_pct = cand['probability'] * 100
                        cand_text = _escape_html(cand['text'])
                        # Check if this candidate is the selected one (using is_selected field or token_id comparison)
                        is_selected = cand.get('is_selected', False) or cand['token_id'] == selected_id
                        marker = "→" if is_selected else " "
                        tooltip_parts.append(f"{marker}{i+1}. '{cand_text}' ({prob_pct:.1f}%)")
                    tooltip = "&#10;".join(tooltip_parts)
                    html_parts.append(f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 3px; cursor: help;" title="{tooltip}">{escaped_text}</span>')
                else:
                    html_parts.append(f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 3px;">{escaped_text}</span>')
                
                color_idx += 1
            
            html_parts.append('</div>')
            return "".join(html_parts)
        
        def _generate_advanced_html(all_token_details):
            """
            Generate detailed HTML for the advanced output panel with fresh and elegant styling
            """
            html_parts = []
            html_parts.append('<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 13px;">')
            
            for sample_info in all_token_details:
                sample_idx = sample_info.get('sample_index', 0)
                token_details = sample_info.get('token_details', [])
                
                # Sample header - 淡雅的青色
                html_parts.append(f'<div style="margin: 16px 0 10px 0; padding: 6px 16px; background: #e0f7fa; color: #006064; border-radius: 20px; font-weight: 600; display: inline-block; font-size: 13px; border: 1px solid #b2ebf2;">Sample {sample_idx + 1}</div>')
                
                # Table with inline styles
                html_parts.append('<div style="max-height: 500px; overflow-y: auto; margin-bottom: 24px; border: 1px solid #f0f0f0; border-radius: 8px;">')
                html_parts.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px; background: white;">')
                
                # Header - 清新淡灰背景，深灰文字
                html_parts.append('<thead><tr style="background: #f8f9fa; border-bottom: 1px solid #e9ecef;">')
                html_parts.append('<th style="padding: 12px 16px; text-align: center; color: #5f6368; font-weight: 600; width: 60px;">#</th>')
                html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600; width: 140px;">Selected</th>')
                html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600;">Top 5 Candidates</th>')
                html_parts.append('</tr></thead><tbody>')
                
                for row_idx, detail in enumerate(token_details):
                    pos = detail.get('position', 0)
                    selected_text = _escape_html(detail.get('selected_token_text', ''))
                    selected_id = detail.get('selected_token_id', -1)
                    candidates = detail.get('top5_candidates', [])
                    
                    # Find selected probability
                    selected_prob = 0
                    non_selected_candidates = []
                    for cand in candidates:
                        is_selected = cand.get('is_selected', False) or cand['token_id'] == selected_id
                        if is_selected:
                            selected_prob = cand['probability'] * 100
                        else:
                            non_selected_candidates.append(cand)
                    
                    # Only show first 5 non-selected candidates with capsule style
                    cand_html_parts = []
                    for idx, cand in enumerate(non_selected_candidates[:5]):
                        prob_pct = cand['probability'] * 100
                        cand_text = _escape_html(cand['text'])
                        # Capsule style with shadow - 清新白底胶囊
                        cand_html_parts.append(
                            f'<span style="display: inline-block; margin: 3px 6px 3px 0; padding: 4px 10px; '
                            f'background: #ffffff; color: #555; '
                            f'border-radius: 12px; font-size: 12px; white-space: nowrap; '
                            f'box-shadow: 0 1px 2px rgba(0,0,0,0.08); border: 1px solid #ebebeb;">'
                            f'<span style="color: #bbb; font-size: 10px; margin-right: 4px;">#{idx+1}</span>'
                            f'{cand_text} <span style="color: #999; font-size: 11px;">({prob_pct:.1f}%)</span></span>'
                        )
                    
                    candidates_html = "".join(cand_html_parts) if cand_html_parts else '<span style="color: #ccc;">-</span>'
                    
                    # Row background alternation - 极其淡的条纹
                    row_bg = "#fbfbfb" if row_idx % 2 == 1 else "white"
                    
                    html_parts.append(f'<tr style="background: {row_bg}; border-bottom: 1px solid #f5f5f5;">')
                    html_parts.append(f'<td style="padding: 10px 16px; text-align: center; color: #9aa0a6; font-size: 12px;">{pos + 1}</td>')
                    
                    # Selected cell - 清新淡绿高亮块
                    html_parts.append(f'<td style="padding: 10px 16px;"><span style="background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 6px; padding: 4px 8px; display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; font-weight: 500;">{selected_text} <span style="opacity: 0.7; font-size: 11px; margin-left: 2px;">{selected_prob:.1f}%</span></span></td>')
                    html_parts.append(f'<td style="padding: 10px 16px; line-height: 1.6;">{candidates_html}</td>')
                    html_parts.append('</tr>')
                
                html_parts.append('</tbody></table></div>')
            
            html_parts.append('</div>')
            return "".join(html_parts)
        
        def inference_cb(
            data_dir_inf_, out_dir_inf_,
            prompt_, num_samples_, max_new_tokens_,
            temperature_, top_k_, dtype_inf_, device_inf_, seed_inf_
        ):
            cache = None
            prompt_tokens = None  # Will hold prompt tokenization info
            try:
                print("🚀 Single model inference started")
                
                # Ensure numeric conversions are robust
                num_samples_int = int(float(num_samples_))
                max_new_tokens_int = int(float(max_new_tokens_))
                temperature_float = float(temperature_)
                top_k_int = int(float(top_k_)) if top_k_ is not None and str(top_k_).strip() != "" else None
                seed_inf_int = int(float(seed_inf_))

                # Get cache instance for subsequent cleanup
                cache = ModelCache()
                
                # Try to tokenize the prompt for display
                try:
                    meta_path = os.path.join(data_dir_inf_, 'meta.pkl')
                    if os.path.exists(meta_path):
                        import pickle
                        with open(meta_path, 'rb') as f:
                            meta = pickle.load(f)
                        old2new_mapping = meta.get('old2new_mapping') or meta.get('old2new')
                        tokenizer_type = meta.get('tokenizer_type') or meta.get('tokenizer', 'char_level')
                        
                        if tokenizer_type == 'custom_json' and old2new_mapping:
                            tokenizer_path = Path.cwd() / "assets" / "tokenizer.json"
                            if tokenizer_path.exists():
                                from tokenizers import Tokenizer
                                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                                prompt_tokens = tokenize_user_input(tokenizer, prompt_, old2new_mapping)
                        elif tokenizer_type == 'gpt2' and old2new_mapping:
                            import tiktoken
                            enc = tiktoken.get_encoding("gpt2")
                            # Simple tokenization for gpt2
                            ids = enc.encode(prompt_, allowed_special={"<|endoftext|>"})
                            prompt_tokens = []
                            for orig_id in ids:
                                try:
                                    decoded_text = enc.decode([orig_id])
                                except:
                                    decoded_text = f"<token_{orig_id}>"
                                mapped_id = old2new_mapping.get(orig_id, orig_id)
                                in_vocab = orig_id in old2new_mapping
                                prompt_tokens.append({
                                    'text': decoded_text,
                                    'original_id': orig_id,
                                    'mapped_id': mapped_id,
                                    'in_vocab': in_vocab
                                })
                        else:
                            # Character level tokenization
                            stoi = meta.get('stoi', {})
                            prompt_tokens = []
                            for ch in prompt_:
                                in_vocab = ch in stoi
                                prompt_tokens.append({
                                    'text': ch,
                                    'original_id': stoi.get(ch, -1),
                                    'mapped_id': stoi.get(ch, -1),
                                    'in_vocab': in_vocab
                                })
                except Exception as e:
                    print(f"Warning: Failed to tokenize prompt for display: {e}")
                    prompt_tokens = None
                
                # Use cached inference function with detailed info enabled
                gen = cached_generate_text(
                    data_dir=data_dir_inf_, out_dir=out_dir_inf_,
                    prompt=prompt_,
                    num_samples=num_samples_int,
                    max_new_tokens=max_new_tokens_int,
                    temperature=temperature_float,
                    top_k=top_k_int,
                    seed=seed_inf_int,
                    device=device_inf_,
                    dtype=dtype_inf_,
                    compile_model=DEFAULT_CONFIG["inference"]["compile_model"],
                    auto_clear_cache=False,
                    return_detailed_info=True  # Enable detailed token info
                )
                
                # Collect token info for highlighting and advanced output
                current_sample_tokens = []
                all_samples_info = []
                all_token_details = []
                current_sample_idx = 0
                prompt_displayed = False
                full_text_output = ""
                
                for item in gen:
                    text_piece, token_detail = item
                    
                    # Check for sample header
                    if text_piece.startswith("Sample ") and text_piece.endswith(":\n"):
                        # Save previous sample if exists
                        if current_sample_tokens:
                            all_samples_info.append({
                                'sample_idx': current_sample_idx - 1,
                                'prompt': prompt_,
                                'tokens': current_sample_tokens.copy()
                            })
                        current_sample_tokens = []
                        prompt_displayed = False
                        current_sample_idx = int(text_piece.replace("Sample ", "").replace(":\n", ""))
                        full_text_output += text_piece
                    elif text_piece == prompt_ and not prompt_displayed:
                        # This is the prompt
                        prompt_displayed = True
                        full_text_output += text_piece
                    elif text_piece.startswith("\n" + "-" * 30):
                        # Separator between samples
                        if current_sample_tokens:
                            all_samples_info.append({
                                'sample_idx': current_sample_idx - 1,
                                'prompt': prompt_,
                                'tokens': current_sample_tokens.copy()
                            })
                        current_sample_tokens = []
                        full_text_output += text_piece
                    else:
                        # Generated token
                        current_sample_tokens.append({
                            'text': text_piece,
                            'token_detail': token_detail
                        })
                        if token_detail:
                            all_token_details.append({
                                'sample_index': current_sample_idx - 1,
                                'token_details': [token_detail]
                            })
                        full_text_output += text_piece
                    
                    # Generate HTML outputs
                    # Main output with token highlighting
                    main_html_parts = []
                    main_html_parts.append('<div style="font-family: system-ui, sans-serif;">')
                    
                    # Show all completed samples
                    for sample_info in all_samples_info:
                        main_html_parts.append(f'<div style="margin-bottom: 15px;"><strong>Sample {sample_info["sample_idx"] + 1}:</strong><br>')
                        main_html_parts.append(_generate_token_html(sample_info['tokens'], prompt_tokens=prompt_tokens))
                        main_html_parts.append('</div>')
                    
                    # Show current sample in progress
                    if current_sample_tokens or prompt_displayed:
                        main_html_parts.append(f'<div style="margin-bottom: 15px;"><strong>Sample {current_sample_idx}:</strong><br>')
                        main_html_parts.append(_generate_token_html(current_sample_tokens, prompt_tokens=prompt_tokens if prompt_displayed else None))
                        main_html_parts.append('</div>')
                    
                    main_html_parts.append('</div>')
                    main_html = "".join(main_html_parts)
                    
                    # Advanced output - consolidate token details by sample
                    consolidated_details = {}
                    for td in all_token_details:
                        s_idx = td['sample_index']
                        if s_idx not in consolidated_details:
                            consolidated_details[s_idx] = {'sample_index': s_idx, 'token_details': []}
                        consolidated_details[s_idx]['token_details'].extend(td['token_details'])
                    
                    advanced_html = _generate_advanced_html(list(consolidated_details.values()))
                    
                    yield main_html, advanced_html
                
                # Final update - add the last sample
                if current_sample_tokens:
                    all_samples_info.append({
                        'sample_idx': current_sample_idx - 1,
                        'prompt': prompt_,
                        'tokens': current_sample_tokens.copy()
                    })
                
                # Final HTML output
                main_html_parts = []
                main_html_parts.append('<div style="font-family: system-ui, sans-serif;">')
                for sample_info in all_samples_info:
                    main_html_parts.append(f'<div style="margin-bottom: 15px;"><strong>Sample {sample_info["sample_idx"] + 1}:</strong><br>')
                    main_html_parts.append(_generate_token_html(sample_info['tokens'], prompt_tokens=prompt_tokens))
                    main_html_parts.append('</div>')
                main_html_parts.append('</div>')
                main_html = "".join(main_html_parts)
                
                consolidated_details = {}
                for td in all_token_details:
                    s_idx = td['sample_index']
                    if s_idx not in consolidated_details:
                        consolidated_details[s_idx] = {'sample_index': s_idx, 'token_details': []}
                    consolidated_details[s_idx]['token_details'].extend(td['token_details'])
                
                advanced_html = _generate_advanced_html(list(consolidated_details.values()))
                
                # Save HTML formatted output to database for persistence across page reloads
                try:
                    ckpt_dir = out_dir_inf_ if out_dir_inf_.endswith('.pt') else os.path.join(out_dir_inf_, 'ckpt.pt')
                    model_dir_for_db = os.path.dirname(ckpt_dir)
                    model_id = dbm.get_model_id_by_dir(model_dir_for_db)
                    if model_id:
                        # Generate plain text content for backward compatibility
                        plain_text_parts = []
                        for sample_info in all_samples_info:
                            sample_text = f"Sample {sample_info['sample_idx'] + 1}:\n"
                            sample_text += prompt_
                            sample_text += "".join([t.get('text', '') for t in sample_info['tokens']])
                            plain_text_parts.append(sample_text)
                        plain_text = "\n\n".join(plain_text_parts)
                        
                        dbm.save_inference_history(model_id, plain_text, main_html, advanced_html)
                        print(f"💾 Inference history saved to database (model_id={model_id})")
                except Exception as save_err:
                    print(f"Warning: Failed to save inference history to database: {save_err}")
                
                yield main_html, advanced_html
                
                print("✅ Single model inference completed successfully")
            
            except UnknownTokenError as e:
                error_msg = f"❌ Error: {str(e)}<br><br>Inference aborted. Please ensure your prompt only contains characters/tokens present in the training vocabulary."
                print(f"❌ Unknown token error: {e}")
                error_html = f'<div style="color: red; padding: 10px; background: #ffe6e6; border-radius: 8px;">{error_msg}</div>'
                yield error_html, ""
                    
            except Exception as e:
                import traceback
                error_msg = f"Error during inference: {str(e)}"
                print(f"❌ Single inference error: {error_msg}")
                print(traceback.format_exc())
                error_html = f'<div style="color: red; padding: 10px; background: #ffe6e6; border-radius: 8px;">{_escape_html(error_msg)}</div>'
                yield error_html, ""
                
            finally:
                # Unified cache cleanup
                try:
                    if cache is None:
                        cache = ModelCache()
                    cache.clear_cache()
                    print("🧹 Single inference completed, cache cleared for optimal performance")
                except Exception as cleanup_error:
                    print(f"Warning: Cache cleanup failed: {cleanup_error}")

        inf_btn.click(
            fn=inference_cb,
            inputs=[data_dir_inf, out_dir_inf, prompt_box,
                    num_samples_box, max_new_tokens_box,
                    temperature_box, top_k_box, dtype_box_inf, device_box_inf, seed_box_inf],
            outputs=[inf_output, inf_advanced_output]
        )

        # ------------------------------------------------------------------
        # Call backs: model selection, reset, delete
        def _reset_updates():
            def _d(val=""): return gr.update(value=val)
            def _b(val=False): return gr.update(value=val) # For boolean checkboxes
            d_train = DEFAULT_CONFIG["training"]
            d_inf = DEFAULT_CONFIG["inference"]
            
            # 获取默认学习率调度器类型的相关组件状态
            default_scheduler = d_train["lr_scheduler_type"]
            warmup_update, lr_decay_update, min_lr_update, step_size_update, step_gamma_update, polynomial_power_update = update_lr_scheduler_params(default_scheduler)
            
            # 获取默认自注意力参数状态
            default_use_self_attention = d_train["use_self_attention"]
            self_attn_updates = update_self_attention_params(default_use_self_attention)
            
            # 原始的返回值
            base_updates = [
                _b(True), _d("new_model"),      # new_model_chk, model_name_box
                _d(), _d(),                     # data_dir_box (train), out_dir_box (train)
                _d(d_train["plot_interval"]), _d(d_train["log_interval"]),
                _d(d_train["num_eval_seeds"]),
                _b(d_train["save_best_val_checkpoint"]),
                _d(d_train["init_from"]),
                _d(d_train["gradient_accumulation_steps"]),
                _d(d_train["batch_size"]), _d(d_train["block_size"]),
                _d(d_train["n_layer"]), _d(d_train["n_head"]), _d(d_train["n_embd"]),
                _d(d_train["dropout"]), _b(d_train["bias"]),
                _d(d_train["learning_rate"]), _d(d_train["max_iters"]), _d(d_train["weight_decay"]),
                _d(d_train["beta1"]), _d(d_train["beta2"]),
                _d(d_train["lr_scheduler_type"]), # 设置默认学习率调度器类型
                # 使用update_lr_scheduler_params的返回值而不是直接设置
                warmup_update,  # warmup_box
                lr_decay_update, # lr_decay_box
                min_lr_update,  # min_lr_box
                step_size_update, # step_size_box
                step_gamma_update, # step_gamma_box
                polynomial_power_update, # polynomial_power_box
                _d(d_train["backend"]), _d(d_train["device"]), _d(d_train["dtype"]),
                _b(d_train["compile_model"]),
                _d(d_train["seed"]), _d(d_train["save_interval"]),
                # Self-attention parameters
                _b(d_train["use_self_attention"]),  # use_self_attention_box
                self_attn_updates[0],  # ffn_hidden_mult_box
                self_attn_updates[1],  # qkv_bias_box
                self_attn_updates[2],  # attn_dropout_box
                self_attn_updates[3],  # resid_dropout_box
                self_attn_updates[4],  # ln_eps_box
                self_attn_updates[5],  # init_std_box
                self_attn_updates[6],  # use_flash_attn_box
                self_attn_updates[7],  # pos_encoding_type_box
                self_attn_updates[8],  # rope_base_box
                # New optimized parameters with default values
                self_attn_updates[9],   # rope_cache_size_box
                self_attn_updates[10],  # alibi_bias_scale_box
                self_attn_updates[11],  # ffn_activation_box
                self_attn_updates[12],  # attention_scale_factor_box
                self_attn_updates[13],  # gradient_checkpointing_box
                self_attn_updates[14],  # cache_strategy_box
                self_attn_updates[15],  # max_cache_size_box
                self_attn_updates[16],  # strict_validation_box
                self_attn_updates[17],  # fallback_on_error_box
                generate_loss_chart_html([], []),        # train_plot (HTML)
                "",                   # train_log (string for HTML box)
                _d(), _d(),           # data_dir_inf, out_dir_inf
                _d(d_inf["prompt"]),
                _d(d_inf["num_samples"]), _d(d_inf["max_new_tokens"]),
                _d(d_inf["temperature"]), _d(d_inf["top_k"]),
                _d(d_inf["dtype"]), # dtype_box_inf
                _d(d_inf["device"]), # device_box_inf
                _d(d_inf["seed"]), # seed_box_inf
                gr.update(), # inf_btn
                "",                 # inf_output
                "",                 # inf_advanced_output
                gr.update(value=[]),  # chatbot - reset to empty chat history
                gr.update(value=DEFAULT_CONFIG["sft"]["system_prompt"]),  # inf_system_prompt - reset to default
                "",  # chat_advanced_output - reset to empty
            ]
            
            # 对比页面组件的重置
            comparison_updates = [
                gr.update(), # comp_left_model
                gr.update(), # comp_right_model
                gr.update(), # comp_left_params
                gr.update(), # comp_right_params
                gr.update(), # comp_left_plot
                gr.update(), # comp_right_plot
                gr.update(), # comp_left_history
                gr.update(), # comp_right_history
                # 左侧模型参数
                _d(d_inf["num_samples"]), # comp_left_num_samples
                _d(d_inf["max_new_tokens"]), # comp_left_max_tokens
                _d(d_inf["temperature"]), # comp_left_temperature
                _d(d_inf["top_k"]), # comp_left_top_k
                _d(d_inf["dtype"]), # comp_left_dtype
                _d(d_inf["seed"]), # comp_left_seed
                # 右侧模型参数
                _d(d_inf["num_samples"]), # comp_right_num_samples
                _d(d_inf["max_new_tokens"]), # comp_right_max_tokens
                _d(d_inf["temperature"]), # comp_right_temperature
                _d(d_inf["top_k"]), # comp_right_top_k
                _d(d_inf["dtype"]), # comp_right_dtype
                _d(d_inf["seed"]), # comp_right_seed
                _d(d_inf["prompt"]), # comp_prompt
                gr.update(), # comp_generate_btn (不重置)
                gr.update(value=""), # comp_left_output
                gr.update(value=""), # comp_right_output
                # Hidden comparison fields
                _d(), # comp_left_data_dir
                _d(), # comp_left_out_dir
                _d(), # comp_right_data_dir
                _d()  # comp_right_out_dir
            ]
            
            return base_updates + comparison_updates

        def select_model_cb(sel: str):
            if not sel:
                return _reset_updates()
            try:
                mid = int(sel.split(" - ")[0])
            except ValueError: # Handle cases where sel might not be in "id - name" format
                return _reset_updates()

            cfg = dbm.get_training_config(mid) or {}
            icfg = dbm.get_inference_config(mid) or {}
            info = dbm.get_model_basic_info(mid) or {}
            name = info.get("name", "unknown_model") # Default name

            # 使用相对路径处理，提高项目移植性
            if "dir_path" in info:
                # 数据库中存储的是相对路径，可以直接使用
                out_dir_root = info["dir_path"]
                # 从存储的路径中提取文件夹名用于数据目录
                folder = os.path.basename(out_dir_root)
                data_processed_dir = os.path.join("data", folder, "processed")
            else:
                # 兼容性处理：如果没有dir_path，使用传统方式
                folder_name_part = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in name)
                folder = f"{folder_name_part}_{mid}"
                data_processed_dir = os.path.join("data", folder, "processed")
                out_dir_root = os.path.join("out", folder)

            def _cfg(k, default_val_from_const): return cfg.get(k, default_val_from_const)
            def _ic(k, default_val_from_const): return icfg.get(k, default_val_from_const)

            loss_log_path = dbm.get_training_log_path(mid)
            loss_plot_html_content = _create_plot_html_from_log(loss_log_path)
            
            train_log_s = ""
            if loss_log_path and os.path.exists(loss_log_path):
                try:
                    with open(loss_log_path, 'rb') as f:
                        log_data_dict = pickle.load(f)
                    
                    log_tr_steps = log_data_dict.get('train_plot_steps', [])
                    log_tr_losses = log_data_dict.get('train_plot_losses', [])
                    log_val_steps = log_data_dict.get('val_plot_steps', [])
                    log_val_losses = log_data_dict.get('val_plot_losses', [])
                    
                    log_lines = []
                    if log_tr_steps:
                        log_lines.append(f"Training history for model {mid} - {name}:")
                        val_map = dict(zip(log_val_steps, log_val_losses)) if log_val_steps and log_val_losses else {}
                        for i, (step, loss) in enumerate(zip(log_tr_steps, log_tr_losses)):
                            line = f"Step {step}: train_loss={loss:.4f}"
                            if step in val_map:
                                line += f", val_loss={val_map[step]:.4f}"
                            log_lines.append(line)
                            if i >= 199: # Limit to 200 lines (0-199)
                                log_lines.append(f"... (showing first 200 of {len(log_tr_steps)} records)")
                                break
                    train_log_s = "\n".join(log_lines)
                except Exception as e_log:
                    train_log_s = f"Error loading training log details: {str(e_log)}"
            
            d_train_defaults = DEFAULT_CONFIG["training"]
            d_inf_defaults = DEFAULT_CONFIG["inference"]

            # Get full inference history including HTML formatted output
            inference_history_data = dbm.get_inference_history_full(mid)
            # Prefer HTML content if available, otherwise fall back to plain text wrapped in basic HTML
            if inference_history_data.get('html_content'):
                inference_history_html = inference_history_data['html_content']
            elif inference_history_data.get('content'):
                # Wrap plain text in basic HTML formatting for display
                plain_text = inference_history_data['content']
                escaped_text = plain_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
                inference_history_html = f'<div style="font-family: monospace; white-space: pre-wrap; line-height: 1.6; padding: 10px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0;">{escaped_text}</div>'
            else:
                inference_history_html = ""
            
            # Get chat history for chat mode (now returns dict with history, advanced_html, system_prompt)
            chat_history_data = dbm.get_chat_history(mid)
            chat_history = chat_history_data.get('history', [])
            chat_advanced_html_saved = chat_history_data.get('advanced_html', '')
            chat_system_prompt_saved = chat_history_data.get('system_prompt', '')
            
            # Get advanced HTML if available
            inference_advanced_html = inference_history_data.get('advanced_html', '')
            
            # 获取模型的学习率调度器类型
            scheduler_type = _cfg("lr_scheduler_type", d_train_defaults["lr_scheduler_type"])
            
            # 根据学习率调度器类型更新相关组件状态
            warmup_update, lr_decay_update, min_lr_update, step_size_update, step_gamma_update, polynomial_power_update = update_lr_scheduler_params(scheduler_type)
            
            # 获取模型的自注意力使用状态并更新相关组件
            use_self_attention = _cfg("use_self_attention", d_train_defaults["use_self_attention"])
            self_attn_updates = update_self_attention_params(use_self_attention)
            
            # 基本组件更新列表
            base_updates = [
                gr.update(value=False),  # new_model_chk
                gr.update(value=name),   # model_name_box
                gr.update(value=data_processed_dir), # data_dir_box (train tab)
                gr.update(value=out_dir_root),       # out_dir_box (train tab)
                gr.update(value=_cfg("plot_interval", d_train_defaults["plot_interval"])),
                gr.update(value=_cfg("log_interval", d_train_defaults["log_interval"])),
                gr.update(value=_cfg("num_eval_seeds", d_train_defaults["num_eval_seeds"])),
                gr.update(value=bool(_cfg("save_best_val_checkpoint", d_train_defaults["save_best_val_checkpoint"]))),
                gr.update(value=_cfg("init_from", d_train_defaults["init_from"])),
                gr.update(value=_cfg("gradient_accumulation_steps", d_train_defaults["gradient_accumulation_steps"])),
                gr.update(value=_cfg("batch_size", d_train_defaults["batch_size"])),
                gr.update(value=_cfg("block_size", d_train_defaults["block_size"])),
                gr.update(value=_cfg("n_layer", d_train_defaults["n_layer"])),
                gr.update(value=_cfg("n_head", d_train_defaults["n_head"])),
                gr.update(value=_cfg("n_embd", d_train_defaults["n_embd"])),
                gr.update(value=_cfg("dropout", d_train_defaults["dropout"])),
                gr.update(value=bool(_cfg("bias", d_train_defaults["bias"]))),
                gr.update(value=_cfg("learning_rate", d_train_defaults["learning_rate"])),
                gr.update(value=_cfg("max_iters", d_train_defaults["max_iters"])),
                gr.update(value=_cfg("weight_decay", d_train_defaults["weight_decay"])),
                gr.update(value=_cfg("beta1", d_train_defaults["beta1"])),
                gr.update(value=_cfg("beta2", d_train_defaults["beta2"])),
                gr.update(value=scheduler_type), # 设置学习率调度器类型
                # 根据学习率调度器类型，设置相关组件的值和交互状态
                warmup_update,  # warmup_box
                lr_decay_update, # lr_decay_box
                min_lr_update,  # min_lr_box
                step_size_update, # step_size_box
                step_gamma_update, # step_gamma_box
                polynomial_power_update, # polynomial_power_box
                gr.update(value=_cfg("backend", d_train_defaults["backend"])),
                gr.update(value=_cfg("device", d_train_defaults["device"])),
                gr.update(value=_cfg("dtype", d_train_defaults["dtype"])),
                gr.update(value=bool(_cfg("compile_model", d_train_defaults["compile_model"]))),
                gr.update(value=_cfg("seed", d_train_defaults["seed"])),
                gr.update(value=_cfg("save_interval", d_train_defaults["save_interval"])),
                # Self-attention parameters
                gr.update(value=use_self_attention),  # use_self_attention_box
                self_attn_updates[0] if use_self_attention else gr.update(visible=False, value=_cfg("ffn_hidden_mult", d_train_defaults["ffn_hidden_mult"])),
                self_attn_updates[1] if use_self_attention else gr.update(visible=False, value=_cfg("qkv_bias", d_train_defaults["qkv_bias"])),
                self_attn_updates[2] if use_self_attention else gr.update(visible=False, value=_cfg("attn_dropout", d_train_defaults["attn_dropout"])),
                self_attn_updates[3] if use_self_attention else gr.update(visible=False, value=_cfg("resid_dropout", d_train_defaults["resid_dropout"])),
                self_attn_updates[4] if use_self_attention else gr.update(visible=False, value=_cfg("ln_eps", d_train_defaults["ln_eps"])),
                self_attn_updates[5] if use_self_attention else gr.update(visible=False, value=_cfg("init_std", d_train_defaults["init_std"])),
                self_attn_updates[6] if use_self_attention else gr.update(visible=False, value=_cfg("use_flash_attn", d_train_defaults["use_flash_attn"])),
                self_attn_updates[7] if use_self_attention else gr.update(visible=False, value=_cfg("pos_encoding_type", d_train_defaults["pos_encoding_type"])),
                self_attn_updates[8] if use_self_attention else gr.update(visible=False, value=_cfg("rope_base", d_train_defaults["rope_base"])),
                # New optimized parameters - with enhanced default handling
                self_attn_updates[9] if use_self_attention else gr.update(visible=False, value=_cfg("rope_cache_size", d_train_defaults["rope_cache_size"])),
                self_attn_updates[10] if use_self_attention else gr.update(visible=False, value=_cfg("alibi_bias_scale", d_train_defaults["alibi_bias_scale"])),
                self_attn_updates[11] if use_self_attention else gr.update(visible=False, value=_cfg("ffn_activation", d_train_defaults["ffn_activation"])),
                self_attn_updates[12] if use_self_attention else gr.update(visible=False, value=_cfg("attention_scale_factor", d_train_defaults["attention_scale_factor"])),
                self_attn_updates[13] if use_self_attention else gr.update(visible=False, value=_cfg("gradient_checkpointing", d_train_defaults["gradient_checkpointing"])),
                self_attn_updates[14] if use_self_attention else gr.update(visible=False, value=_cfg("cache_strategy", d_train_defaults["cache_strategy"])),
                self_attn_updates[15] if use_self_attention else gr.update(visible=False, value=_cfg("max_cache_size", d_train_defaults["max_cache_size"])),
                self_attn_updates[16] if use_self_attention else gr.update(visible=False, value=_cfg("strict_validation", d_train_defaults["strict_validation"])),
                self_attn_updates[17] if use_self_attention else gr.update(visible=False, value=_cfg("fallback_on_error", d_train_defaults["fallback_on_error"])),
                loss_plot_html_content,        # train_plot (HTML)
                train_log_s,                   # train_log (string for HTML box)
                gr.update(value=data_processed_dir), # data_dir_inf (infer tab)
                gr.update(value=out_dir_root),       # out_dir_inf (infer tab)
                gr.update(value=_ic("prompt", d_inf_defaults["prompt"])),
                gr.update(value=_ic("num_samples", d_inf_defaults["num_samples"])),
                gr.update(value=_ic("max_new_tokens", d_inf_defaults["max_new_tokens"])),
                gr.update(value=_ic("temperature", d_inf_defaults["temperature"])),
                gr.update(value=_ic("top_k", d_inf_defaults["top_k"])),
                gr.update(value=_ic("dtype", d_inf_defaults["dtype"])), # dtype_box_inf
                gr.update(value=_ic("device", d_inf_defaults["device"])), # device_box_inf
                gr.update(value=_ic("seed", d_inf_defaults["seed"])), # seed_box_inf
                gr.update(), # inf_btn (添加缺失的组件)
                inference_history_html, # inf_output (HTML formatted)
                inference_advanced_html, # inf_advanced_output (advanced token info)
                # Chat mode components
                gr.update(value=chat_history),  # chatbot - load saved chat history
                gr.update(value=chat_system_prompt_saved if chat_system_prompt_saved is not None else ''),  # inf_system_prompt - load saved system prompt (empty string is valid)
                chat_advanced_html_saved,  # chat_advanced_output - load saved advanced HTML
            ]
            
            # 对比页面组件更新，但不直接更新左右模型选择框，自选
            comparison_updates = [
                gr.update(), # comp_left_model
                gr.update(), # comp_right_model
                gr.update(), # comp_left_params
                gr.update(), # comp_right_params
                gr.update(), # comp_left_plot
                gr.update(), # comp_right_plot
                gr.update(), # comp_left_history
                gr.update(), # comp_right_history
                # 左侧模型参数
                gr.update(value=_ic("num_samples", d_inf_defaults["num_samples"])), # comp_left_num_samples
                gr.update(value=_ic("max_new_tokens", d_inf_defaults["max_new_tokens"])), # comp_left_max_tokens
                gr.update(value=_ic("temperature", d_inf_defaults["temperature"])), # comp_left_temperature
                gr.update(value=_ic("top_k", d_inf_defaults["top_k"])), # comp_left_top_k
                gr.update(value=_ic("dtype", d_inf_defaults["dtype"])), # comp_left_dtype
                gr.update(value=_ic("seed", d_inf_defaults["seed"])), # comp_left_seed
                # 右侧模型参数
                gr.update(value=_ic("num_samples", d_inf_defaults["num_samples"])), # comp_right_num_samples
                gr.update(value=_ic("max_new_tokens", d_inf_defaults["max_new_tokens"])), # comp_right_max_tokens
                gr.update(value=_ic("temperature", d_inf_defaults["temperature"])), # comp_right_temperature
                gr.update(value=_ic("top_k", d_inf_defaults["top_k"])), # comp_right_top_k
                gr.update(value=_ic("dtype", d_inf_defaults["dtype"])), # comp_right_dtype
                gr.update(value=_ic("seed", d_inf_defaults["seed"])), # comp_right_seed
                gr.update(value=_ic("prompt", d_inf_defaults["prompt"])), # comp_prompt
                gr.update(), # comp_generate_btn
                gr.update(), # comp_left_output
                gr.update(), # comp_right_output
                # Hidden comparison fields
                gr.update(value=""), # comp_left_data_dir
                gr.update(value=""), # comp_left_out_dir
                gr.update(value=""), # comp_right_data_dir
                gr.update(value="")  # comp_right_out_dir
            ]
            
            return base_updates + comparison_updates

        outputs_for_model_select_and_delete = [
            new_model_chk, model_name_box,
            data_dir_box, out_dir_box,
            plot_interval_box, log_interval_box,
            num_eval_seeds_box, save_best_val_ckpt_box, init_from_box,
            grad_acc_box, batch_size_box, block_size_box,
            n_layer_box, n_head_box, n_embd_box,
            dropout_box, bias_box,
            lr_box, max_iters_box, weight_decay_box,
            beta1_box, beta2_box,
            lr_scheduler_box, warmup_box,
            lr_decay_box, min_lr_box,
            step_size_box, step_gamma_box, polynomial_power_box,
            backend_box, device_box, dtype_box, compile_box,
            seed_box, save_interval_box,
            # Self-attention parameters
            use_self_attention_box, ffn_hidden_mult_box, qkv_bias_box, attn_dropout_box,
            resid_dropout_box, ln_eps_box, init_std_box, use_flash_attn_box, pos_encoding_type_box,
            rope_base_box,
            # New optimized parameters - ADD THESE MISSING ONES
            rope_cache_size_box, alibi_bias_scale_box, ffn_activation_box, attention_scale_factor_box,
            gradient_checkpointing_box, cache_strategy_box, max_cache_size_box,
            strict_validation_box, fallback_on_error_box,
            train_plot, train_log,
            data_dir_inf, out_dir_inf,
            prompt_box, num_samples_box, max_new_tokens_box,
            temperature_box, top_k_box, dtype_box_inf, device_box_inf, seed_box_inf,
            inf_btn, inf_output, inf_advanced_output,
            # Chat mode - add chatbot, system_prompt and advanced_output for loading saved data
            chatbot, inf_system_prompt, chat_advanced_output,
            # 添加对比页面组件
            comp_left_model, comp_right_model,
            comp_left_params, comp_right_params,
            comp_left_plot, comp_right_plot,
            comp_left_history, comp_right_history,
            comp_left_num_samples, comp_left_max_tokens,
            comp_left_temperature, comp_left_top_k, comp_left_dtype, comp_left_seed,
            comp_right_num_samples, comp_right_max_tokens, 
            comp_right_temperature, comp_right_top_k, comp_right_dtype, comp_right_seed,
            comp_prompt, comp_generate_btn,
            comp_left_output, comp_right_output,
            # ADD MISSING HIDDEN FIELDS
            comp_left_data_dir, comp_left_out_dir, comp_right_data_dir, comp_right_out_dir
        ]
        model_dropdown.change(
            fn=select_model_cb,
            inputs=[model_dropdown],
            outputs=outputs_for_model_select_and_delete
        )

        def delete_model_cb(sel: str):
            if sel and " - " in sel:
                try:
                    model_id = int(sel.split(" - ")[0])
                    dbm.delete_model(model_id)
                except Exception as e:
                    print(f"Error deleting model: {e}") # Log error
            
            # 获取更新后的模型列表
            updated_choices = _get_model_choices_list()
            
            # 更新主下拉框和对比页面的两个下拉框
            main_dropdown_update = gr.update(choices=updated_choices, value=None)
            comp_left_update = gr.update(choices=updated_choices, value=None)
            comp_right_update = gr.update(choices=updated_choices, value=None)
            
            # 重置所有组件为默认值
            reset_values = _reset_updates()
            
            # 额外清空对比页面特定的内容（在reset_updates基础上进一步确保清空）
            # 找到对比页面组件在reset_values中的位置并确保它们被清空
            # 由于_reset_updates已经处理了大部分重置，我们只需要确保对比页面的特殊组件被清空
            
            return [main_dropdown_update, comp_left_update, comp_right_update] + reset_values


        delete_model_btn.click(
            fn=delete_model_cb,
            inputs=[model_dropdown],
            # 更新输出列表，包括主下拉框和对比页面的两个下拉框，然后是其他重置组件
            outputs=[model_dropdown, comp_left_model, comp_right_model] + outputs_for_model_select_and_delete 
        )

        refresh_models_btn.click(
            lambda: [gr.update(choices=_get_model_choices_list()) for _ in range(3)],
            [],
            [model_dropdown, comp_left_model, comp_right_model]
        )

        # ------------------------------------------------------------------
        # Call backs: language switch
        def switch_language(lang_code: str):
            Tn = LANG_JSON[lang_code]
            
            return [
                gr.update(label=Tn["language_label"], value=lang_code), # lang_select itself
                # Tab labels
                gr.update(label=Tn["data_process_tab"]), gr.update(label=Tn["train_tab"]), gr.update(label=Tn["infer_tab"]), gr.update(label=Tn["compare_tab"]),
                # Top bar
                gr.update(label=Tn["registered_models"]), gr.update(value=Tn["refresh_tables"]), gr.update(value=Tn["delete_selected_model"]),
                # Model management (Data Processing Tab)
                gr.update(label=Tn["new_model"]), gr.update(label=Tn["model_name"]),
                # Data processing panel
                gr.update(label=Tn["dp_paste_text"]), gr.update(label=Tn["dp_txt_dir"]),
                gr.update(label=Tn["dp_no_val_set"]), gr.update(label=Tn["dp_use_gpt2_tokenizer"]),
                gr.update(label=Tn["dp_train_split"]), gr.update(label=Tn["dp_num_proc"]),
                gr.update(value=Tn["dp_start_btn"]), gr.update(label=Tn["dp_result"]),
                # Training panel
                gr.update(value=f"### {Tn['train_params_title']}"), # train_params_title_md
                gr.update(label=Tn["train_data_dir"]), gr.update(label=Tn["train_out_dir"]),
                gr.update(label=Tn["train_backend"]), gr.update(label=Tn["train_device"]), gr.update(label=Tn["train_dtype"]),
                gr.update(label=Tn["train_compile_model"]),
                gr.update(label=Tn["train_eval_interval"]), gr.update(label=Tn["train_log_interval"]),
                gr.update(label=Tn["train_num_eval_seeds"]), gr.update(label=Tn["train_save_best_val_ckpt"]),
                gr.update(label=Tn["train_init_from"]), gr.update(label=Tn["train_seed"]),
                gr.update(label=Tn["train_gas"]), gr.update(label=Tn["train_batch_size"]), gr.update(label=Tn["train_block_size"]),
                gr.update(label=Tn["train_n_layer"]), gr.update(label=Tn["train_n_head"]), gr.update(label=Tn["train_n_embd"]),
                gr.update(label=Tn["train_dropout"]), gr.update(label=Tn["train_bias"]),
                gr.update(label=Tn["train_lr"]), gr.update(label=Tn["train_max_iters"]), gr.update(label=Tn["train_weight_decay"]),
                gr.update(label=Tn["train_beta1"]), gr.update(label=Tn["train_beta2"]),
                gr.update(label=Tn["train_lr_scheduler"]),
                gr.update(label=Tn["train_warmup_iters"]), gr.update(label=Tn["train_lr_decay_iters"]), gr.update(label=Tn["train_min_lr"]),
                gr.update(label=Tn["train_step_size"]), gr.update(label=Tn["train_step_gamma"]),
                gr.update(label=Tn["train_poly_power"]),
                gr.update(label=Tn["train_save_interval"]),
                gr.update(value=Tn["train_start_btn"]), gr.update(value=Tn["stop_btn"]),
                gr.update(label=Tn["train_log"]), gr.update(label=Tn["train_plot"]),
                # Inference panel
                gr.update(label=Tn["dp_processed_dir"]), gr.update(label=Tn["inf_out_dir"]),
                gr.update(label=Tn["inf_prompt"]), gr.update(label=Tn["inf_num_samples"]),
                gr.update(label=Tn["inf_max_new_tokens"]), gr.update(label=Tn["inf_temperature"]),
                gr.update(label=Tn["inf_top_k"]), gr.update(label=Tn["inf_dtype"]), 
                gr.update(label=Tn["inf_device"]), gr.update(label=Tn["inf_seed"]), # seed_box_inf label
                gr.update(value=Tn["inf_start_btn"]), gr.update(label=Tn["inf_result"]),
                gr.update(label=Tn["inf_advanced_output"]),  # inf_advanced_output
                gr.update(label=Tn["inf_advanced_output"]),  # advanced_accordion
                # Chat mode components
                gr.update(label=Tn["inf_chat_mode"]),  # inf_chat_mode
                gr.update(label=Tn["inf_chat_history"]),  # chatbot
                gr.update(label=Tn["inf_system_prompt"]),  # inf_system_prompt
                gr.update(label=Tn["inf_user_input"]),  # inf_user_input
                gr.update(value=Tn["inf_send_btn"]),  # inf_send_btn
                gr.update(value=Tn["inf_clear_chat"]),  # inf_clear_btn
                gr.update(label=Tn["inf_chat_advanced"]),  # chat_advanced_accordion
                gr.update(label=Tn["inf_advanced_output"]),  # chat_advanced_output
                # Comparison tab
                gr.update(label=Tn["compare_left_model"]), gr.update(label=Tn["compare_right_model"]),
                gr.update(label=Tn["compare_model_params"]), gr.update(label=Tn["compare_model_params"]),
                gr.update(label=Tn["compare_loss_curve"]), gr.update(label=Tn["compare_loss_curve"]),
                gr.update(label=Tn["compare_inference_history"]), gr.update(label=Tn["compare_inference_history"]),
                # 左侧模型参数
                gr.update(label=Tn["inf_num_samples"]), gr.update(label=Tn["inf_max_new_tokens"]), 
                gr.update(label=Tn["inf_temperature"]), gr.update(label=Tn["inf_top_k"]), gr.update(label=Tn["inf_dtype"]), gr.update(label=Tn["inf_seed"]),
                # 右侧模型参数
                gr.update(label=Tn["inf_num_samples"]), gr.update(label=Tn["inf_max_new_tokens"]), 
                gr.update(label=Tn["inf_temperature"]), gr.update(label=Tn["inf_top_k"]), gr.update(label=Tn["inf_dtype"]), gr.update(label=Tn["inf_seed"]),
                gr.update(label=Tn["compare_shared_prompt"]), gr.update(value=Tn["compare_generate_btn"]),
                gr.update(label=Tn["compare_left_output"]), gr.update(label=Tn["compare_right_output"]),
                # Self-attention parameters
                gr.update(label=Tn["train_self_attn_title"]),
                gr.update(label=Tn["train_use_self_attention"]),  # use_self_attention_box
                gr.update(label=Tn["train_ffn_hidden_mult"]),    # ffn_hidden_mult_box
                gr.update(label=Tn["train_qkv_bias"]),           # qkv_bias_box
                gr.update(label=Tn["train_attn_dropout"]),       # attn_dropout_box
                gr.update(label=Tn["train_resid_dropout"]),      # resid_dropout_box
                gr.update(label=Tn["train_ln_eps"]),             # ln_eps_box
                gr.update(label=Tn["train_init_std"]),           # init_std_box
                gr.update(label=Tn["train_use_flash_attn"]),     # use_flash_attn_box
                gr.update(label=Tn["train_pos_encoding_type"]),  # pos_encoding_type_box
                gr.update(label=Tn["train_rope_base"]),          # rope_base_box
                # New optimized parameters
                gr.update(label=Tn["train_rope_cache_size"]),
                gr.update(label=Tn["train_alibi_bias_scale"]),
                gr.update(label=Tn["train_ffn_activation"]),
                gr.update(label=Tn["train_attention_scale_factor"]),
                gr.update(label=Tn["train_gradient_checkpointing"]),
                gr.update(label=Tn["train_cache_strategy"]),
                gr.update(label=Tn["train_max_cache_size"]),
                gr.update(label=Tn["train_strict_validation"]),
                gr.update(label=Tn["train_fallback_on_error"]),
            ]

        lang_select_outputs = [
            lang_select,
            data_process_tab, train_tab, inf_tab, comp_tab,
            model_dropdown, refresh_models_btn, delete_model_btn,
            new_model_chk, model_name_box,
            input_text, txt_dir, no_val_set, use_gpt2,
            train_split, num_proc, process_btn, process_output,
            train_params_title_md, data_dir_box, out_dir_box,
            backend_box, device_box, dtype_box, compile_box,
            plot_interval_box, log_interval_box, num_eval_seeds_box,
            save_best_val_ckpt_box, init_from_box, seed_box,
            grad_acc_box, batch_size_box, block_size_box,
            n_layer_box, n_head_box, n_embd_box,
            dropout_box, bias_box, lr_box, max_iters_box, weight_decay_box,
            beta1_box, beta2_box, lr_scheduler_box,
            warmup_box, lr_decay_box, min_lr_box, step_size_box, step_gamma_box, polynomial_power_box,
            save_interval_box, train_btn, stop_btn,
            train_log, train_plot,
            data_dir_inf, out_dir_inf, prompt_box,
            num_samples_box, max_new_tokens_box, temperature_box, top_k_box,
            dtype_box_inf, device_box_inf, seed_box_inf,
            inf_btn, inf_output, inf_advanced_output, advanced_accordion,
            # Chat mode components
            inf_chat_mode, chatbot, inf_system_prompt, inf_user_input, 
            inf_send_btn, inf_clear_btn, chat_advanced_accordion,
            chat_advanced_output,
            # Comparison tab components
            comp_left_model, comp_right_model,
            comp_left_params, comp_right_params,
            comp_left_plot, comp_right_plot,
            comp_left_history, comp_right_history,
            comp_left_num_samples, comp_left_max_tokens, 
            comp_left_temperature, comp_left_top_k, comp_left_dtype, comp_left_seed,
            comp_right_num_samples, comp_right_max_tokens, 
            comp_right_temperature, comp_right_top_k, comp_right_dtype, comp_right_seed,
            comp_prompt, comp_generate_btn,
            comp_left_output, comp_right_output,
            # Self-attention parameters  
            self_attn_accordion, use_self_attention_box, ffn_hidden_mult_box, qkv_bias_box, attn_dropout_box,
            resid_dropout_box, ln_eps_box, init_std_box, use_flash_attn_box, 
            pos_encoding_type_box, rope_base_box,
            # New optimized parameters
            rope_cache_size_box, alibi_bias_scale_box, ffn_activation_box, attention_scale_factor_box,
            gradient_checkpointing_box, cache_strategy_box, max_cache_size_box,
            strict_validation_box, fallback_on_error_box
        ]

        lang_select.change(
            fn=switch_language,
            inputs=[lang_select],
            outputs=lang_select_outputs # Use the defined list
        )

        # Initialize LR scheduler params display logic on app load
        demo.load(
            fn=lambda: update_lr_scheduler_params(DEFAULT_CONFIG["training"]["lr_scheduler_type"]),
            inputs=None,
            outputs=[
                warmup_box, lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box
            ],
            queue=False
        )
        
        # 手动切换学习率调度器类型时也更新相关组件
        lr_scheduler_box.change(
            fn=update_lr_scheduler_params,
            inputs=[lr_scheduler_box],
            outputs=[
                warmup_box, lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box
            ],
            queue=False
        )
        
        # ------------------------------------------------------------------ #
        # Call backs: comparison page
        # ------------------------------------------------------------------ #
        def select_model_for_comparison_cb(sel: str, is_left: bool):
            """
            Select model for comparison (left or right side)
            """
            if not sel:
                return [{}, generate_loss_chart_html([], []), "", "", ""]
            
            try:
                mid = int(sel.split(" - ")[0])
            except ValueError:
                return [{}, generate_loss_chart_html([], []), "", "", ""]
        
            # Get model info
            cfg = dbm.get_training_config(mid) or {}
            icfg = dbm.get_inference_config(mid) or {}
            info = dbm.get_model_basic_info(mid) or {}
            name = info.get("name", "unknown_model")
            
            # 使用相对路径处理，提高项目移植性
            if "dir_path" in info:
                # 数据库中存储的是相对路径，可以直接使用
                out_dir_root = info["dir_path"]
                # 从存储的路径中提取文件夹名用于数据目录
                folder = os.path.basename(out_dir_root)
                data_processed_dir = os.path.join("data", folder, "processed")
            else:
                # 兼容性处理：如果没有dir_path，使用传统方式
                folder_name_part = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in name)
                folder = f"{folder_name_part}_{mid}"
                data_processed_dir = os.path.join("data", folder, "processed")
                out_dir_root = os.path.join("out", folder)

            # Generate loss curve
            loss_log_path = dbm.get_training_log_path(mid)
            loss_plot_html_content = _create_plot_html_from_log(loss_log_path)
            
            # Get inference history if any
            inference_history = dbm.get_inference_history(mid) or ""
            
            # Create parameter display dictionary - only include the most important parameters
            display_params = {}
            if cfg:
                try:
                    display_params = {
                        "Model Structure": {
                            "layers": cfg.get("n_layer"),
                            "heads": cfg.get("n_head"),
                            "embedding_dim": cfg.get("n_embd"),
                            "block_size": cfg.get("block_size"),
                            "dropout": cfg.get("dropout"),
                            "bias": cfg.get("bias")
                        },
                        "Training": {
                            "learning_rate": cfg.get("learning_rate"),
                            "batch_size": cfg.get("batch_size"),
                            "iterations": cfg.get("max_iters"),
                            "scheduler": cfg.get("lr_scheduler_type")
                        }
                    }
                    
                    # Add self-attention parameters if enabled
                    if cfg.get("use_self_attention", False):
                        display_params["Self-Attention"] = {
                            "ffn_hidden_mult": cfg.get("ffn_hidden_mult"),
                            "qkv_bias": cfg.get("qkv_bias"),
                            "attn_dropout": cfg.get("attn_dropout"),
                            "resid_dropout": cfg.get("resid_dropout"),
                            "ln_eps": cfg.get("ln_eps"),
                            "init_std": cfg.get("init_std"),
                            "use_flash_attn": cfg.get("use_flash_attn"),
                            "pos_encoding_type": cfg.get("pos_encoding_type"),
                            "rope_base": cfg.get("rope_base")
                        }
                except Exception as e:
                    print(f"Error formatting parameters: {e}")
            
            return [display_params, loss_plot_html_content, inference_history, data_processed_dir, out_dir_root]
        
        # Left model selection
        comp_left_model.change(
            fn=lambda sel: select_model_for_comparison_cb(sel, True),
            inputs=[comp_left_model],
            outputs=[comp_left_params, comp_left_plot, comp_left_history, comp_left_data_dir, comp_left_out_dir]
        )
        
        # Right model selection
        comp_right_model.change(
            fn=lambda sel: select_model_for_comparison_cb(sel, False),
            inputs=[comp_right_model],
            outputs=[comp_right_params, comp_right_plot, comp_right_history, comp_right_data_dir, comp_right_out_dir]
        )
        
        def dual_inference_cb(
            left_data_dir, left_out_dir,
            right_data_dir, right_out_dir,
            prompt,
            left_num_samples, left_max_tokens, left_temperature, left_top_k, left_dtype, left_seed,
            right_num_samples, right_max_tokens, right_temperature, right_top_k, right_dtype, right_seed
        ):
            """
            Optimized dual model concurrent inference using caching system and improved concurrency strategy
            """
            print("🔥 Starting dual model comparison inference...")
            
            if not left_out_dir or not right_out_dir:
                error_msg = "Please select two models for comparison first."
                return error_msg, error_msg
            
            if not prompt.strip():
                error_msg = "Prompt is empty, please enter starting text."
                return error_msg, error_msg
            
            # Initialize output
            left_output = ""
            right_output = ""
            cache = None
            
            try:
                # Parameter validation and conversion
                try:
                    left_params = {
                        'num_samples': int(float(left_num_samples)),
                        'max_tokens': int(float(left_max_tokens)),
                        'temperature': float(left_temperature),
                        'top_k': int(float(left_top_k)) if left_top_k is not None and str(left_top_k).strip() != "" else None,
                        'seed': int(float(left_seed)),
                        'dtype': left_dtype
                    }
                except ValueError as e:
                    error_msg = f"Left model parameter error: {str(e)}"
                    yield error_msg, right_output
                    return
                    
                try:
                    right_params = {
                        'num_samples': int(float(right_num_samples)),
                        'max_tokens': int(float(right_max_tokens)),
                        'temperature': float(right_temperature),
                        'top_k': int(float(right_top_k)) if right_top_k is not None and str(right_top_k).strip() != "" else None,
                        'seed': int(float(right_seed)),
                        'dtype': right_dtype
                    }
                except ValueError as e:
                    error_msg = f"Right model parameter error: {str(e)}"
                    yield left_output, error_msg
                    return
                
                # Get cache instance to optimize model loading
                cache = ModelCache()
                cache_info = cache.get_cache_info()
                print(f"🔍 Comparison inference started - Cache status: {cache_info}")
                
                # Verify checkpoint files exist and get model info
                left_ckpt_path = left_out_dir if left_out_dir.endswith('.pt') else os.path.join(left_out_dir, 'ckpt.pt')
                right_ckpt_path = right_out_dir if right_out_dir.endswith('.pt') else os.path.join(right_out_dir, 'ckpt.pt')
                
                if not os.path.exists(left_ckpt_path):
                    error_msg = f"Left model checkpoint not found: {left_ckpt_path}"
                    yield error_msg, right_output
                    return
                    
                if not os.path.exists(right_ckpt_path):
                    error_msg = f"Right model checkpoint not found: {right_ckpt_path}"
                    yield left_output, error_msg
                    return
                
                print(f"✅ Both checkpoint files verified")
                
                # Pre-check model types and compatibility
                try:
                    left_checkpoint = torch.load(left_ckpt_path, map_location='cpu')
                    left_model_args = left_checkpoint['model_args']
                    left_model_type = cache._detect_model_type(left_model_args)
                    
                    right_checkpoint = torch.load(right_ckpt_path, map_location='cpu')
                    right_model_args = right_checkpoint['model_args']
                    right_model_type = cache._detect_model_type(right_model_args)
                    
                    print(f"🔍 Model types detected - Left: {left_model_type}, Right: {right_model_type}")
                    
                    # Check vocab size compatibility for comparison
                    left_vocab_size = left_model_args.get('vocab_size', 0)
                    right_vocab_size = right_model_args.get('vocab_size', 0)
                    
                    if left_vocab_size != right_vocab_size:
                        print(f"⚠️ Warning: Different vocab sizes - Left: {left_vocab_size}, Right: {right_vocab_size}")
                        
                    # Test model loading capability
                    try:
                        left_model, left_gptconf, left_encode, left_decode = cache.get_model_and_meta(
                            left_ckpt_path, left_data_dir, 'cpu', left_dtype
                        )
                        print(f"✅ Left model ({left_model_type}) loaded successfully")
                        
                        right_model, right_gptconf, right_encode, right_decode = cache.get_model_and_meta(
                            right_ckpt_path, right_data_dir, 'cpu', right_dtype
                        )
                        print(f"✅ Right model ({right_model_type}) loaded successfully")
                        
                        # Clear CPU models to free memory before actual inference
                        del left_model, right_model
                        import gc
                        gc.collect()
                        
                    except Exception as model_load_error:
                        error_msg = f"Model loading test failed: {str(model_load_error)}"
                        print(f"❌ {error_msg}")
                        yield error_msg, error_msg
                        return
                        
                except Exception as e:
                    print(f"⚠️ Warning: Model compatibility check failed: {e}")
                    # Continue anyway, let the inference attempt reveal specific issues
                
                # Smart device allocation - estimate memory requirements and assign optimal devices
                left_memory_req = 0
                right_memory_req = 0
                
                if os.path.exists(left_ckpt_path):
                    left_size_mb = os.path.getsize(left_ckpt_path) / (1024 * 1024)
                    left_memory_req = device_manager.estimate_model_memory(left_size_mb)
                
                if os.path.exists(right_ckpt_path):
                    right_size_mb = os.path.getsize(right_ckpt_path) / (1024 * 1024)
                    right_memory_req = device_manager.estimate_model_memory(right_size_mb)
                
                # Allocate optimal device combination for dual models
                left_device, right_device = device_manager.allocate_devices_for_comparison(
                    left_memory_req, right_memory_req
                )
                
                print(f"🎯 Device allocation: left model={left_device}, right model={right_device}")
                
                # Use improved queues and more efficient thread pool
                left_queue = queue.Queue(maxsize=1000)  # Increase queue capacity
                right_queue = queue.Queue(maxsize=1000)
                
                # Use thread pool executor for better performance
                with ThreadPoolExecutor(max_workers=2, thread_name_prefix="ModelInference") as executor:
                    
                    def run_cached_inference(data_dir, out_dir, params, result_queue, model_name, assigned_device):
                        """Wrapper function for running cached inference with smart device allocation"""
                        try:
                            gen = cached_generate_text(
                                data_dir=data_dir,
                                out_dir=out_dir,
                                prompt=prompt,
                                num_samples=params['num_samples'],
                                max_new_tokens=params['max_tokens'],
                                temperature=params['temperature'],
                                top_k=params['top_k'],
                                seed=params['seed'],
                                device=assigned_device,  # Use intelligently allocated device
                                dtype=params['dtype'],
                                compile_model=DEFAULT_CONFIG["inference"]["compile_model"],
                                auto_clear_cache=False  # Disable auto cleanup in comparison inference, do unified cleanup instead
                            )
                            
                            # Batch process generated text fragments to reduce queue operations
                            text_buffer = ""
                            buffer_size = 5  # Batch output every 5 characters
                            
                            for piece in gen:
                                text_buffer += piece
                                if len(text_buffer) >= buffer_size:
                                    result_queue.put(('data', text_buffer))
                                    text_buffer = ""
                            
                            # Output remaining content
                            if text_buffer:
                                result_queue.put(('data', text_buffer))
                            
                            result_queue.put(('done', None))
                        
                        except UnknownTokenError as e:
                            # Handle unknown token errors with user-friendly message
                            error_msg = f"❌ {model_name} Error: {str(e)}\n\nInference aborted. Please ensure your prompt only contains characters/tokens present in the training vocabulary."
                            print(f"❌ {model_name} Unknown token error: {str(e)}")
                            result_queue.put(('error', error_msg))
                            result_queue.put(('done', None))
                            
                        except Exception as e:
                            error_msg = f"{model_name} generation error: {str(e)}"
                            print(f"❌ {error_msg}")
                            import traceback
                            print(traceback.format_exc())
                            result_queue.put(('error', error_msg))
                            result_queue.put(('done', None))
                    
                    # Submit two inference tasks using intelligently allocated devices
                    left_future = executor.submit(
                        run_cached_inference, 
                        left_data_dir, left_out_dir, left_params, left_queue, "Left model", left_device
                    )
                    
                    right_future = executor.submit(
                        run_cached_inference, 
                        right_data_dir, right_out_dir, right_params, right_queue, "Right model", right_device
                    )
                    
                    # Track completion status
                    left_done = False
                    right_done = False
                    last_yield_time = 0
                    min_yield_interval = 0.05  # Minimum output interval to reduce UI update frequency
                    
                    # Improved concurrent output processing
                    while not (left_done and right_done):
                        current_time = time.time()
                        updated = False
                        
                        # Batch process left model output
                        left_batch = []
                        while not left_done:
                            try:
                                msg_type, data = left_queue.get_nowait()
                                if msg_type == 'data':
                                    left_batch.append(data)
                                elif msg_type == 'error':
                                    left_output = data
                                    updated = True
                                    break
                                elif msg_type == 'done':
                                    left_done = True
                                    break
                            except queue.Empty:
                                break
                        
                        if left_batch:
                            left_output += ''.join(left_batch)
                            updated = True
                        
                        # Batch process right model output
                        right_batch = []
                        while not right_done:
                            try:
                                msg_type, data = right_queue.get_nowait()
                                if msg_type == 'data':
                                    right_batch.append(data)
                                elif msg_type == 'error':
                                    right_output = data
                                    updated = True
                                    break
                                elif msg_type == 'done':
                                    right_done = True
                                    break
                            except queue.Empty:
                                break
                        
                        if right_batch:
                            right_output += ''.join(right_batch)
                            updated = True
                        
                        # Control output frequency to reduce UI pressure
                        if updated and (current_time - last_yield_time) >= min_yield_interval:
                            yield left_output, right_output
                            last_yield_time = current_time
                        elif not updated:
                            # Wait a bit when no updates to avoid high CPU usage
                            time.sleep(0.02)  # Reduce wait time
                    
                    # Wait for tasks to complete
                    try:
                        left_future.result(timeout=10.0)  # Increased timeout
                        right_future.result(timeout=10.0)
                        print("✅ Both inference tasks completed")
                    except concurrent.futures.TimeoutError:
                        print("⚠️ Warning: Some inference tasks may not have completed properly")
                    except Exception as e:
                        print(f"⚠️ Warning: Task completion error: {e}")
                    
                    # Final output
                    yield left_output, right_output
                    
                    print("🏁 Comparison inference completed successfully")
                    
            except Exception as e:
                error_msg = f"Comparison inference error: {str(e)}"
                print(f"❌ {error_msg}")
                import traceback
                print(traceback.format_exc())
                
                # If either output is empty, put error message
                if not left_output.strip():
                    left_output = error_msg
                if not right_output.strip():
                    right_output = error_msg
                    
                yield left_output, right_output
                
            finally:
                # Unified cache cleanup - ensure resources are released after comparison inference
                try:
                    if cache is None:
                        cache = ModelCache()
                    cache.clear_cache()
                    print("🧹 Comparison inference completed, cache cleared for optimal performance")
                except Exception as cleanup_error:
                    print(f"Warning: Cache cleanup failed: {cleanup_error}")
        
        # Connect the generate button to the dual inference callback
        comp_generate_btn.click(
            fn=dual_inference_cb,
            inputs=[
                comp_left_data_dir, comp_left_out_dir,
                comp_right_data_dir, comp_right_out_dir,
                comp_prompt,
                comp_left_num_samples, comp_left_max_tokens, comp_left_temperature, comp_left_top_k, comp_left_dtype, comp_left_seed,
                comp_right_num_samples, comp_right_max_tokens, comp_right_temperature, comp_right_top_k, comp_right_dtype, comp_right_seed
            ],
            outputs=[comp_left_output, comp_right_output]
        )
        # ------------------------------------------------------------------
        # SFT Callbacks
        # ------------------------------------------------------------------
        
        def sft_refresh_models():
            choices = _get_model_choices_list()
            return gr.update(choices=choices, value=None)

        sft_refresh_model_btn.click(
            fn=sft_refresh_models,
            inputs=[],
            outputs=[sft_base_model]
        )
        
        def sft_load_dataset(file_obj, dir_path):
            current_lang = lang_select.value
            T_current = LANG_JSON[current_lang]
            
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
            
        sft_validate_btn.click(
            fn=sft_load_dataset,
            inputs=[sft_dataset_file, sft_dataset_dir],
            outputs=[sft_format_status, sft_dataset_state]
        )
        
        def sft_train_cb(
            model_selection, dataset, epochs, lr, batch_size, 
            max_seq_len, grad_acc, warmup_ratio, system_prompt
        ):
            current_lang = lang_select.value
            T_current = LANG_JSON[current_lang]
            
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
            
            # Generate dummy plot data
            empty_plot = generate_loss_chart_html([], [])
            
            if not model_selection or " - " not in model_selection:
                yield f"<div style='color:red;'>❌ Please select a base model</div>", "", empty_plot
                return
                
            model_id = int(model_selection.split(" - ")[0])
            model_info = dbm.get_model(model_id)
            if not model_info:
                yield f"<div style='color:red;'>❌ Model not found</div>", "", empty_plot
                return
                
            base_ckpt_path = os.path.join(model_info['out_dir'], 'ckpt.pt')
            
            if not dataset:
                yield f"<div style='color:red;'>{T_current['sft_no_dataset']}</div>", "", empty_plot
                return
                
            # Create SFT output directory
            sft_out_dir = os.path.join(model_info['out_dir'], 'sft')
            os.makedirs(sft_out_dir, exist_ok=True)
            
            yield make_progress_html(0, 100), "🚀 Starting SFT Training...", empty_plot
            
            try:
                generator = sft_train_generator(
                    base_model_ckpt_path=base_ckpt_path,
                    data_dir=model_info['processed_data_dir'], # Not strictly used but passed for compat
                    dataset=dataset,
                    out_dir=sft_out_dir,
                    epochs=int(epochs),
                    learning_rate=lr,
                    batch_size=int(batch_size),
                    max_seq_length=int(max_seq_len),
                    gradient_accumulation_steps=int(grad_acc),
                    warmup_ratio=warmup_ratio,
                    system_prompt=system_prompt
                )
                
                for progress_html, log_msg, plot_data in generator:
                    # Plot data format: (steps, losses, val_steps, val_losses)
                    if plot_data and len(plot_data) >= 2:
                         # Convert steps/losses to lists of tuples for chart generator
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

        sft_start_btn.click(
            fn=sft_train_cb,
            inputs=[
                sft_base_model, sft_dataset_state,
                sft_epochs, sft_learning_rate, sft_batch_size,
                sft_max_seq_length, sft_gradient_accumulation,
                sft_warmup_ratio, sft_system_prompt
            ],
            outputs=[sft_progress, sft_log, sft_plot]
        )
        
        def sft_stop_cb():
            stop_sft_training()
            return "🛑 Stopping SFT..."
            
        sft_stop_btn.click(
            fn=sft_stop_cb,
            inputs=[],
            outputs=[sft_log]
        )

        # ------------------------------------------------------------------
        # Chat Callbacks
        # ------------------------------------------------------------------
        
        # Helper functions for chat token display
        def _generate_user_tokenization_html(tokens_info):
            """Generate HTML to display user input tokenization - inline with message"""
            if not tokens_info:
                return ""
            
            html_parts = []
            html_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 3px; margin-top: 6px; padding-top: 6px; border-top: 1px dashed #ccc;">')
            html_parts.append('<span style="font-size: 11px; color: #666; margin-right: 4px;">📝 Tokens:</span>')
            
            for i, token_info in enumerate(tokens_info):
                text = _escape_html(token_info['text'])
                orig_id = token_info['original_id']
                mapped_id = token_info['mapped_id']
                in_vocab = token_info['in_vocab']
                
                color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
                border_color = "#4caf50" if in_vocab else "#f44336"
                border_style = "1px solid " + border_color
                
                tooltip = f"Token #{i+1}&#10;Text: '{text}'&#10;Original ID: {orig_id}&#10;Mapped ID: {mapped_id}&#10;In Vocab: {'Yes' if in_vocab else 'No'}"
                
                html_parts.append(f'''<span style="background-color: {color}; padding: 1px 4px; border-radius: 3px; 
                    border: {border_style}; cursor: help; font-size: 11px;" title="{tooltip}">{text}</span>''')
            
            html_parts.append('</div>')
            return "".join(html_parts)
        
        def _generate_response_html_with_tokens(response_tokens):
            """Generate HTML for response text with token highlighting"""
            if not response_tokens:
                return ""
            
            html_parts = []
            for i, token_info in enumerate(response_tokens):
                text = _escape_html(token_info['text'])
                color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
                detail = token_info.get('token_detail')
                
                if detail:
                    candidates = detail.get('top5_candidates', [])
                    tooltip_parts = [f"#{detail.get('position', i)+1}: '{text}'"]
                    for j, cand in enumerate(candidates[:5]):
                        prob_pct = cand['probability'] * 100
                        cand_text = _escape_html(cand['text'])
                        marker = "→" if cand['token_id'] == detail.get('selected_token_id') else " "
                        tooltip_parts.append(f"{marker}{j+1}. '{cand_text}' ({prob_pct:.1f}%)")
                    tooltip = "&#10;".join(tooltip_parts)
                else:
                    tooltip = f"Token: {text}"
                
                html_parts.append(f'''<span style="background-color: {color}; padding: 0px 2px; border-radius: 2px; 
                    cursor: help;" title="{tooltip}">{text}</span>''')
            
            return "".join(html_parts)
        
        def _generate_chat_advanced_html(all_token_details, response_tokens, system_prompt_tokens=None):
            """Generate detailed HTML for chat advanced output panel - matching non-chat mode styling"""
            html_parts = []
            html_parts.append('<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 13px;">')
            
            # System prompt tokenization section (only shown once per conversation start or when provided)
            if system_prompt_tokens:
                html_parts.append('<div style="margin: 16px 0 10px 0; padding: 6px 16px; background: #fff3e0; color: #e65100; border-radius: 20px; font-weight: 600; display: inline-block; font-size: 13px; border: 1px solid #ffe0b2;">System Prompt Tokens</div>')
                html_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 4px; margin-bottom: 16px; padding: 12px; background: #fffbf5; border: 1px solid #ffe0b2; border-radius: 8px;">')
                for i, token_info in enumerate(system_prompt_tokens):
                    text = _escape_html(token_info['text'])
                    orig_id = token_info['original_id']
                    mapped_id = token_info['mapped_id']
                    in_vocab = token_info['in_vocab']
                    color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
                    border_color = "#4caf50" if in_vocab else "#f44336"
                    tooltip = f"Token #{i+1}&#10;Text: '{text}'&#10;Original ID: {orig_id}&#10;Mapped ID: {mapped_id}&#10;In Vocab: {'Yes' if in_vocab else 'No'}"
                    html_parts.append(f'''<span style="background-color: {color}; padding: 2px 6px; border-radius: 4px; border: 1px solid {border_color}; cursor: help; font-size: 12px;" title="{tooltip}">{text}</span>''')
                html_parts.append('</div>')
            
            if not all_token_details:
                html_parts.append('</div>')
                return "".join(html_parts)
            
            # Response header - 淡雅的青色 (matching non-chat mode)
            html_parts.append('<div style="margin: 16px 0 10px 0; padding: 6px 16px; background: #e0f7fa; color: #006064; border-radius: 20px; font-weight: 600; display: inline-block; font-size: 13px; border: 1px solid #b2ebf2;">Response Token Details</div>')
            
            # Table with inline styles (matching non-chat mode)
            html_parts.append('<div style="max-height: 500px; overflow-y: auto; margin-bottom: 24px; border: 1px solid #f0f0f0; border-radius: 8px;">')
            html_parts.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px; background: white;">')
            
            # Header - 清新淡灰背景，深灰文字
            html_parts.append('<thead><tr style="background: #f8f9fa; border-bottom: 1px solid #e9ecef;">')
            html_parts.append('<th style="padding: 12px 16px; text-align: center; color: #5f6368; font-weight: 600; width: 60px;">#</th>')
            html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600; width: 140px;">Selected</th>')
            html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600;">Top 5 Candidates</th>')
            html_parts.append('</tr></thead><tbody>')
            
            for row_idx, detail in enumerate(all_token_details):
                pos = detail.get('position', 0)
                selected_text = _escape_html(detail.get('selected_token_text', ''))
                selected_id = detail.get('selected_token_id', -1)
                candidates = detail.get('top5_candidates', [])
                
                # Find selected probability
                selected_prob = 0
                non_selected_candidates = []
                for cand in candidates:
                    is_selected = cand.get('is_selected', False) or cand['token_id'] == selected_id
                    if is_selected:
                        selected_prob = cand['probability'] * 100
                    else:
                        non_selected_candidates.append(cand)
                
                # Only show first 5 non-selected candidates with capsule style
                cand_html_parts = []
                for idx, cand in enumerate(non_selected_candidates[:5]):
                    prob_pct = cand['probability'] * 100
                    cand_text = _escape_html(cand['text'])
                    # Capsule style with shadow - 清新白底胶囊
                    cand_html_parts.append(
                        f'<span style="display: inline-block; margin: 3px 6px 3px 0; padding: 4px 10px; '
                        f'background: #ffffff; color: #555; '
                        f'border-radius: 12px; font-size: 12px; white-space: nowrap; '
                        f'box-shadow: 0 1px 2px rgba(0,0,0,0.08); border: 1px solid #ebebeb;">'
                        f'<span style="color: #bbb; font-size: 10px; margin-right: 4px;">#{idx+1}</span>'
                        f'{cand_text} <span style="color: #999; font-size: 11px;">({prob_pct:.1f}%)</span></span>'
                    )
                
                candidates_html = "".join(cand_html_parts) if cand_html_parts else '<span style="color: #ccc;">-</span>'
                
                # Row background alternation - 极其淡的条纹
                row_bg = "#fbfbfb" if row_idx % 2 == 1 else "white"
                
                html_parts.append(f'<tr style="background: {row_bg}; border-bottom: 1px solid #f5f5f5;">')
                html_parts.append(f'<td style="padding: 10px 16px; text-align: center; color: #9aa0a6; font-size: 12px;">{pos + 1}</td>')
                
                # Selected cell - 清新淡绿高亮块
                html_parts.append(f'<td style="padding: 10px 16px;"><span style="background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 6px; padding: 4px 8px; display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; font-weight: 500;">{selected_text} <span style="opacity: 0.7; font-size: 11px; margin-left: 2px;">{selected_prob:.1f}%</span></span></td>')
                html_parts.append(f'<td style="padding: 10px 16px; line-height: 1.6;">{candidates_html}</td>')
                html_parts.append('</tr>')
            
            html_parts.append('</tbody></table></div>')
            html_parts.append('</div>')
            return "".join(html_parts)
        
        def chat_cb(user_msg, history, model_sel, sys_prompt, max_tokens, temp, top_k, seed, device_val):
            if not user_msg or not user_msg.strip():
                # Show error for empty message
                gr.Warning("⚠️ Please enter a message.")
                return "", history or [], ""
            
            # Update history with user message
            history = history or []
            history.append((user_msg, None))
            yield "", history, ""
            
            # Validate model
            if not model_sel or " - " not in model_sel:
                history[-1] = (user_msg, "❌ Please select a model first.")
                yield "", history, ""
                return

            model_id = int(model_sel.split(" - ")[0])
            model_info = dbm.get_model(model_id)
            if not model_info:
                history[-1] = (user_msg, "❌ Model not found.")
                yield "", history, ""
                return
            
            # Load model (using cache if possible - reuse inference cache logic or simplified)
            # For simplicity, we'll load directly here, but ideally we should use ModelCache
            # Let's use a simplified version since we need streaming reference
            
            try:
                ckpt_path = os.path.join(model_info['out_dir'], 'ckpt.pt')
                sft_ckpt_path = os.path.join(model_info['out_dir'], 'sft', 'ckpt_sft.pt')
                
                # Prefer SFT checkpoint if available
                load_path = sft_ckpt_path if os.path.exists(sft_ckpt_path) else ckpt_path
                
                # Check for tokenizer
                tokenizer_path = Path.cwd() / "assets" / "tokenizer.json"
                if not tokenizer_path.exists():
                     history[-1] = (user_msg, "❌ tokenizer.json not found.")
                     yield "", history, "", ""
                     return
                
                from tokenizers import Tokenizer
                tokenizer = Tokenizer.from_file(str(tokenizer_path))
                
                # Load meta.pkl to get old2new mapping for token ID remapping
                # model_info['processed_data_dir'] points to data/{model_name}/processed
                processed_data_dir = model_info.get('processed_data_dir', '')
                meta_path = os.path.join(processed_data_dir, 'meta.pkl') if processed_data_dir else None
                old2new_mapping = None
                new2old_mapping = None
                
                if meta_path and os.path.exists(meta_path):
                    import pickle
                    with open(meta_path, 'rb') as f:
                        meta = pickle.load(f)
                    # Try both possible key names for compatibility
                    old2new_mapping = meta.get('old2new_mapping') or meta.get('old2new')
                    if old2new_mapping:
                        new2old_mapping = {new_id: old_id for old_id, new_id in old2new_mapping.items()}
                
                if old2new_mapping is None:
                    history[-1] = (user_msg, f"❌ Error: Token ID mapping not found. meta_path={meta_path}, exists={os.path.exists(meta_path) if meta_path else False}. Please ensure the model was trained with the custom tokenizer.")
                    yield "", history, "", ""
                    return
                
                # Load saved token IDs from database (if available) to avoid re-tokenization issues
                chat_history_data = dbm.get_chat_history(model_id)
                saved_token_ids = chat_history_data.get('token_ids', [])
                
                # Build history_token_ids from saved data
                # This contains all previous conversation context in token form
                history_token_ids = None
                if saved_token_ids:
                    # Use the last saved all_token_ids as the history context
                    # This preserves exact token boundaries
                    if saved_token_ids and 'all_token_ids' in saved_token_ids[-1]:
                        history_token_ids = saved_token_ids[-1]['all_token_ids']
                
                # Tokenize user input for display
                user_tokens_info = tokenize_user_input(tokenizer, user_msg, old2new_mapping)
                user_tokenization_html = _generate_user_tokenization_html(user_tokens_info)
                
                # Tokenize system prompt for display (only for first message or when changed)
                system_prompt_tokens = tokenize_user_input(tokenizer, sys_prompt, old2new_mapping) if sys_prompt else []
                
                # Load model
                checkpoint = torch.load(load_path, map_location=device_val)
                model_args = checkpoint['model_args']
                
                # Determine model type
                is_self_attention_model = any(key in model_args for key in [
                    'ffn_hidden_mult', 'qkv_bias', 'attn_dropout', 'resid_dropout'
                ])
                
                if is_self_attention_model:
                    gptconf = GPTSelfAttnConfig(**model_args)
                    model = GPTSelfAttn(gptconf)
                else:
                    gptconf = GPTConfig(**model_args)
                    model = GPT(gptconf)
                    
                state_dict = checkpoint['model']
                unwanted_prefix = '_orig_mod.'
                for k in list(state_dict.keys()):
                    if k.startswith(unwanted_prefix):
                        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
                model.load_state_dict(state_dict)
                model.to(device_val)
                model.eval()
                
                # Prepare messages from history - extract plain text for model
                messages = []
                for user_entry, bot_entry in history[:-1]: # Skip last pending
                    # User entry might be HTML with tokenization, extract plain text
                    if isinstance(user_entry, str):
                        # Remove HTML tags if present to get plain text for model
                        import re
                        plain_user = re.sub(r'<[^>]+>', '', user_entry)
                        # Also remove the "📝 Tokens:" prefix if present
                        plain_user = re.sub(r'📝 Tokens:.*', '', plain_user, flags=re.DOTALL).strip()
                    else:
                        plain_user = str(user_entry)
                    
                    messages.append({"role": "user", "content": plain_user})
                    if bot_entry:
                        # Bot entry might also be HTML with token colors
                        if isinstance(bot_entry, str):
                            plain_bot = re.sub(r'<[^>]+>', '', bot_entry)
                        else:
                            plain_bot = str(bot_entry)
                        messages.append({"role": "assistant", "content": plain_bot})
                
                messages.append({"role": "user", "content": user_msg})
                
                # Set random seed for reproducibility
                torch.manual_seed(int(seed))
                if 'cuda' in device_val:
                    torch.cuda.manual_seed(int(seed))
                
                # Generate with proper token ID mappings and detailed info
                # Pass history_token_ids to avoid re-tokenization of conversation history
                generator = chat_generate(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    system_prompt=sys_prompt,
                    max_new_tokens=int(max_tokens),
                    temperature=temp,
                    top_k=int(top_k),
                    old2new_mapping=old2new_mapping,
                    new2old_mapping=new2old_mapping,
                    return_detailed_info=True,  # Enable detailed token info
                    history_token_ids=history_token_ids  # Use saved token IDs to avoid re-tokenization
                )
                
                all_token_details = []
                response_tokens = []
                final_token_data = None  # Will store the final token IDs for saving
                
                # Create user message HTML with tokenization
                user_msg_html = f'<div style="font-family: system-ui, sans-serif;">{_escape_html(user_msg)}{user_tokenization_html}</div>'
                
                for item in generator:
                    text_piece, token_detail = item
                    
                    # Check if this is the final message with token IDs
                    if token_detail and token_detail.get('is_final'):
                        final_token_data = token_detail
                        continue
                    
                    # Skip empty text pieces (shouldn't happen but just in case)
                    if not text_piece:
                        continue
                    
                    # Collect token info
                    response_tokens.append({
                        'text': text_piece,
                        'token_detail': token_detail
                    })
                    if token_detail:
                        all_token_details.append(token_detail)
                    
                    # Generate response HTML with token highlighting
                    response_html = f'<div style="font-family: system-ui, sans-serif;">{_generate_response_html_with_tokens(response_tokens)}</div>'
                    
                    history[-1] = (user_msg_html, response_html)
                    
                    # Generate advanced HTML (include system prompt tokens only for first turn)
                    show_sys_tokens = system_prompt_tokens if len(history) == 1 else None
                    advanced_html = _generate_chat_advanced_html(all_token_details, response_tokens, show_sys_tokens)
                    
                    yield "", history, advanced_html
                
                # Final update
                response_html = f'<div style="font-family: system-ui, sans-serif;">{_generate_response_html_with_tokens(response_tokens)}</div>'
                history[-1] = (user_msg_html, response_html)
                show_sys_tokens = system_prompt_tokens if len(history) == 1 else None
                advanced_html = _generate_chat_advanced_html(all_token_details, response_tokens, show_sys_tokens)
                
                # Update saved token IDs with the new conversation turn
                if final_token_data:
                    saved_token_ids.append({
                        'all_token_ids': final_token_data.get('all_token_ids', []),
                        'generated_token_ids': final_token_data.get('generated_token_ids', []),
                        'prompt_length': final_token_data.get('prompt_length', 0)
                    })
                
                # Save chat history to database for persistence (including token_ids)
                try:
                    dbm.save_chat_history(model_id, history, advanced_html, sys_prompt, saved_token_ids)
                    print(f"💾 Chat history saved to database (model_id={model_id}, token_ids={len(saved_token_ids)} turns)")
                except Exception as save_err:
                    print(f"Warning: Failed to save chat history to database: {save_err}")
                
                yield "", history, advanced_html
                    
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                print(f"Chat error: {error_detail}")
                history[-1] = (user_msg, f"❌ Error: {str(e)}")
                yield "", history, f'<div style="color: red;">Error: {_escape_html(str(e))}</div>'

        inf_send_btn.click(
            fn=chat_cb,
            inputs=[
                inf_user_input, chatbot,
                model_dropdown, 
                inf_system_prompt,
                max_new_tokens_box, temperature_box, top_k_box, seed_box_inf, device_box_inf
            ],
            outputs=[inf_user_input, chatbot, chat_advanced_output]
        )
        
        inf_user_input.submit(
             fn=chat_cb,
            inputs=[
                inf_user_input, chatbot,
                model_dropdown, 
                inf_system_prompt,
                max_new_tokens_box, temperature_box, top_k_box, seed_box_inf, device_box_inf
            ],
            outputs=[inf_user_input, chatbot, chat_advanced_output]
        )
        
        def clear_chat(model_sel):
            # Clear chat history from database if a model is selected
            if model_sel and " - " in model_sel:
                try:
                    model_id = int(model_sel.split(" - ")[0])
                    dbm.clear_chat_history(model_id)
                    print(f"🗑️ Chat history cleared for model_id={model_id}")
                except Exception as e:
                    print(f"Warning: Failed to clear chat history from database: {e}")
            return [], ""
            
        inf_clear_btn.click(fn=clear_chat, inputs=[model_dropdown], outputs=[chatbot, chat_advanced_output])

    return demo

# ----------------- Launch -------------------
if __name__ == "__main__":
    app = build_app_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)