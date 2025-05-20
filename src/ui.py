# src/ui.py
import os
import pickle
# import io # No longer needed for matplotlib buffer for this plot
import numpy as np # Required for generate_loss_chart_html
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
# torch.nn, torch.nn.functional, etc. are kept as they might be used by imported functions
# from torch.optim import AdamW # Not directly used in ui.py
# from torch.nn.parallel import DistributedDataParallel as DDP # Not directly used in ui.py
# from torch.distributed import init_process_group, destroy_process_group # Not directly used in ui.py

import gradio as gr
# from PIL import Image # No longer needed for this plot
# import matplotlib # No longer needed for this plot
# matplotlib.use('Agg') # No longer needed for this plot
# import matplotlib.pyplot as plt # No longer needed for this plot

from src.config import DEFAULT_CONFIG, LANG_JSON # Assuming IntegerTypes was not used in the original snippet
from src.db_manager import DBManager
from src.data_process import process_data
from src.train import train_model_generator, stop_training # train_model_generator assumed to be adapted
from src.infer import generate_text

dbm = DBManager()

# --- SVG Chart Generation Function ---
# (The generate_loss_chart_html function provided by you should be pasted here)
# For brevity, I'm assuming it's correctly defined as in your example.
# Make sure it's at this top level in the file.

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

    # Ensure legend items only appear if data exists
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

def build_app_interface(selected_lang: str = "zh"):
    """
    Top-level UI function
    Implemented:
        · Logic for new model/model name, automatic directory, dropdown refresh/delete
        · Language switching: After switching the `lang_select` dropdown, **all component labels & default values** are refreshed synchronously
        · Dynamic HTML/SVG loss plot
    """

    T = LANG_JSON[selected_lang]

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
            model_dropdown = gr.Dropdown(label=T["registered_models"], choices=_get_model_choices_list(), interactive=True)
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
                    backend_box = gr.Textbox(label=T["train_backend"], value=DEFAULT_CONFIG["training"]["backend"])
                    device_box = gr.Dropdown(label=T["train_device"], choices=["cpu", "cuda"],
                                             value=DEFAULT_CONFIG["training"]["device"])
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
                
                with gr.Row(): # New row for scheduler specific params as per original code
                    step_size_box = gr.Number(label="Step Size", # Assuming T["train_step_size"] if translated
                                               value=DEFAULT_CONFIG["training"]["step_size"])
                    step_gamma_box = gr.Number(label="Step Gamma", # Assuming T["train_step_gamma"]
                                                value=DEFAULT_CONFIG["training"]["step_gamma"])
                    polynomial_power_box = gr.Number(label="Polynomial Power", # Assuming T["train_poly_power"]
                                                      value=DEFAULT_CONFIG["training"]["polynomial_power"])
                    save_interval_box = gr.Number(label=T["train_save_interval"],
                                                   value=DEFAULT_CONFIG["training"]["save_interval"])


                train_btn = gr.Button(T["train_start_btn"])
                stop_btn = gr.Button(T["stop_btn"])

                with gr.Row():
                    with gr.Column(scale=1):
                        train_progress = gr.HTML(label="Training Progress") # Label can be T["train_progress_label"]
                        train_log = gr.HTML(label=T["train_log"], elem_id="train-log-box")
                    with gr.Column(scale=2):
                        train_plot = gr.HTML(label=T["train_plot"]) # Changed from gr.Image

            # -------------- Inference Tab -------------- #
            with gr.Tab(T["infer_tab"]) as inf_tab:
                with gr.Row():
                    data_dir_inf = gr.Textbox(label=T["dp_processed_dir"], value="", interactive=False)
                    out_dir_inf = gr.Textbox(label=T["inf_out_dir"], value="", interactive=False)

                prompt_box = gr.Textbox(label=T["inf_prompt"],
                                          value=DEFAULT_CONFIG["inference"]["prompt"], lines=5)

                with gr.Row():
                    num_samples_box = gr.Number(label=T["inf_num_samples"],
                                                value=DEFAULT_CONFIG["inference"]["num_samples"])
                    max_new_tokens_box = gr.Number(label=T["inf_max_new_tokens"],
                                                   value=DEFAULT_CONFIG["inference"]["max_new_tokens"])
                    temperature_box = gr.Number(label=T["inf_temperature"],
                                                value=DEFAULT_CONFIG["inference"]["temperature"])
                    top_k_box = gr.Number(label=T["inf_top_k"],
                                          value=DEFAULT_CONFIG["inference"]["top_k"])
                    seed_box_inf = gr.Number(label=T["inf_seed"],
                                             value=DEFAULT_CONFIG["inference"]["seed"])

                inf_btn = gr.Button(T["inf_start_btn"])
                inf_output = gr.Textbox(label=T["inf_result"], lines=10, interactive=False)

        # ------------------------------------------------------------------ #
        # Call backs: data processing / training / inference
        # ------------------------------------------------------------------ #
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
                return msg, gr.update(choices=new_choices, value=new_val)
            except Exception as e:
                return f"❌ Error: {str(e)}", gr.update()

        process_btn.click(
            fn=data_processing_cb,
            inputs=[new_model_chk, model_name_box, model_dropdown,
                    input_text, txt_dir,
                    train_split, no_val_set, use_gpt2, num_proc],
            outputs=[process_output, model_dropdown]
        )

        stop_btn.click(fn=stop_training, inputs=[], outputs=[])
        
        # -----------------------------
        # LR Scheduler Callback
        # -----------------------------
        def update_lr_scheduler_params(scheduler_type):
            defaults_train = DEFAULT_CONFIG["training"]
            warmup_update = gr.update(interactive=False, value="") # Use default if not interactive
            lr_decay_update = gr.update(interactive=False, value="")
            min_lr_update = gr.update(interactive=False, value="")
            step_size_update = gr.update(interactive=False, value="")
            step_gamma_update = gr.update(interactive=False, value="")
            polynomial_power_update = gr.update(interactive=False, value="")
            
            if scheduler_type == "none":
                pass # All remain non-interactive and empty/default
            elif scheduler_type == "cosine":
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
            seed_, save_interval_
        ):
            empty_plot_html = generate_loss_chart_html([], [])
            try:
                # Convert Number inputs safely (they might be float-like strings or actual numbers)
                num_eval_seeds_int = int(float(num_eval_seeds_))
                if not (0 <= num_eval_seeds_int <= 2**32 - 1):
                    raise ValueError("num_eval_seeds out of range")
                # Add more specific validation for other numeric inputs if needed.
                # int(float(val)) is a robust way to convert from gr.Number
            except ValueError as e:
                yield (f"<div style='color:red;'>Configuration Error: {str(e)}</div>", str(e), empty_plot_html)
                return

            try:
                defaults_train = DEFAULT_CONFIG["training"]
                def safe_int(v, default_val): return default_val if v == "" or v is None else int(float(v))
                def safe_float(v, default_val): return default_val if v == "" or v is None else float(v)

                # Assuming train_model_generator is adapted to yield:
                # (progress_html, log_text, (train_steps, train_losses, val_steps, val_losses)_or_None)
                gen = train_model_generator(
                    data_dir=data_dir_,
                    out_dir=out_dir_,
                    plot_interval=safe_int(plot_interval_, defaults_train["plot_interval"]),
                    log_interval=safe_int(log_interval_, defaults_train["log_interval"]),
                    num_eval_seeds=num_eval_seeds_int, # Already converted and validated
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
                    polynomial_power=safe_float(polynomial_power_, defaults_train["polynomial_power"]), # Original was float, ensure consistency
                    backend=backend_, device=device_, dtype=dtype_,
                    compile_model=bool(compile_),
                    seed=safe_int(seed_, defaults_train["seed"]),
                    save_interval=safe_int(save_interval_, defaults_train["save_interval"])
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
                yield (f"<div style='color:red;'>{err_msg}</div>", err_msg, empty_plot_html)

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
                seed_box, save_interval_box
            ],
            outputs=[train_progress, train_log, train_plot]
        )

        # ------------------------------------------------------------------ #
        # Call backs: inference
        # ------------------------------------------------------------------ #
        def inference_cb(
            data_dir_inf_, out_dir_inf_,
            prompt_, num_samples_, max_new_tokens_,
            temperature_, top_k_, seed_inf_
        ):
            try:
                output_stream_text = ""
                # Ensure numeric conversions are robust
                num_samples_int = int(float(num_samples_))
                max_new_tokens_int = int(float(max_new_tokens_))
                temperature_float = float(temperature_)
                top_k_int = int(float(top_k_)) if top_k_ is not None and str(top_k_).strip() != "" else None
                seed_inf_int = int(float(seed_inf_))

                gen = generate_text(
                    data_dir=data_dir_inf_, out_dir=out_dir_inf_,
                    prompt=prompt_,
                    num_samples=num_samples_int,
                    max_new_tokens=max_new_tokens_int,
                    temperature=temperature_float,
                    top_k=top_k_int,
                    seed=seed_inf_int,
                    device=DEFAULT_CONFIG["inference"]["device"], # These should ideally be UI configurable too
                    dtype=DEFAULT_CONFIG["inference"]["dtype"],
                    compile_model=DEFAULT_CONFIG["inference"]["compile_model"]
                )
                
                for piece in gen:
                    output_stream_text += piece
                    yield output_stream_text
            except Exception as e:
                import traceback
                print(f"Inference callback error: {traceback.format_exc()}")
                yield f"Error during inference: {str(e)}"

        inf_btn.click(
            fn=inference_cb,
            inputs=[data_dir_inf, out_dir_inf, prompt_box,
                    num_samples_box, max_new_tokens_box,
                    temperature_box, top_k_box, seed_box_inf],
            outputs=inf_output
        )

        # ------------------------------------------------------------------ #
        # Call backs: model selection, reset, delete
        # ------------------------------------------------------------------ #
        def _reset_updates():
            def _d(val=""): return gr.update(value=val)
            def _b(val=False): return gr.update(value=val) # For boolean checkboxes
            d_train = DEFAULT_CONFIG["training"]
            d_inf = DEFAULT_CONFIG["inference"]
            
            # This list must match the `outputs` of model_dropdown.change and delete_model_btn.click
            return [
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
                _d(d_train["lr_scheduler_type"]), # This will trigger update_lr_scheduler_params if changed
                _d(d_train["warmup_iters"]), _d(d_train["lr_decay_iters"]), _d(d_train["min_lr"]),
                _d(d_train["step_size"]), _d(d_train["step_gamma"]),_d(d_train["polynomial_power"]),
                _d(d_train["backend"]), _d(d_train["device"]), _d(d_train["dtype"]),
                _b(d_train["compile_model"]),
                _d(d_train["seed"]), _d(d_train["save_interval"]),
                generate_loss_chart_html([], []), # train_plot (HTML)
                "",                               # train_log (empty string for HTML box)
                _d(), _d(),                       # data_dir_inf, out_dir_inf
                _d(d_inf["prompt"]),
                _d(d_inf["num_samples"]), _d(d_inf["max_new_tokens"]),
                _d(d_inf["temperature"]), _d(d_inf["top_k"]), 
                _d(d_inf["seed"]), # seed_box_inf
                ""                 # inf_output
            ]

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
            
            # Sanitize name for path or use ID for folder to be safer
            # For now, assuming original name_mid logic is acceptable
            folder_name_part = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in name) # Basic sanitize
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

            updates_list = [
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
                gr.update(value=_cfg("lr_scheduler_type", d_train_defaults["lr_scheduler_type"])),
                gr.update(value=_cfg("warmup_iters", d_train_defaults["warmup_iters"])),
                gr.update(value=_cfg("lr_decay_iters", d_train_defaults["lr_decay_iters"])),
                gr.update(value=_cfg("min_lr", d_train_defaults["min_lr"])),
                gr.update(value=_cfg("step_size", d_train_defaults["step_size"])),
                gr.update(value=_cfg("step_gamma", d_train_defaults["step_gamma"])),
                gr.update(value=_cfg("polynomial_power", d_train_defaults["polynomial_power"])),
                gr.update(value=_cfg("backend", d_train_defaults["backend"])),
                gr.update(value=_cfg("device", d_train_defaults["device"])),
                gr.update(value=_cfg("dtype", d_train_defaults["dtype"])),
                gr.update(value=bool(_cfg("compile_model", d_train_defaults["compile_model"]))),
                gr.update(value=_cfg("seed", d_train_defaults["seed"])),
                gr.update(value=_cfg("save_interval", d_train_defaults["save_interval"])),
                loss_plot_html_content,        # train_plot (HTML)
                train_log_s,                   # train_log (string for HTML box)
                gr.update(value=data_processed_dir), # data_dir_inf (infer tab)
                gr.update(value=out_dir_root),       # out_dir_inf (infer tab)
                gr.update(value=_ic("prompt", d_inf_defaults["prompt"])),
                gr.update(value=_ic("num_samples", d_inf_defaults["num_samples"])),
                gr.update(value=_ic("max_new_tokens", d_inf_defaults["max_new_tokens"])),
                gr.update(value=_ic("temperature", d_inf_defaults["temperature"])),
                gr.update(value=_ic("top_k", d_inf_defaults["top_k"])),
                gr.update(value=_ic("seed", d_inf_defaults["seed"])), # seed_box_inf
                dbm.get_inference_history(mid) or "" # inf_output
            ]
            return updates_list

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
            train_plot, train_log,
            data_dir_inf, out_dir_inf,
            prompt_box, num_samples_box, max_new_tokens_box,
            temperature_box, top_k_box, seed_box_inf,
            inf_output
        ]
        model_dropdown.change(
            fn=select_model_cb,
            inputs=[model_dropdown],
            outputs=outputs_for_model_select_and_delete
        )

        def delete_model_cb(sel: str):
            if sel and " - " in sel:
                try:
                    dbm.delete_model(int(sel.split(" - ")[0]))
                except Exception as e:
                    print(f"Error deleting model: {e}") # Log error
            # After deleting, reset UI and refresh choices
            updated_choices = _get_model_choices_list()
            # The first output is model_dropdown itself
            return [gr.update(choices=updated_choices, value=None)] + _reset_updates()


        delete_model_btn.click(
            fn=delete_model_cb,
            inputs=[model_dropdown],
            # The first output is model_dropdown, then the rest are from _reset_updates
            outputs=[model_dropdown] + outputs_for_model_select_and_delete 
        )

        refresh_models_btn.click(lambda: gr.update(choices=_get_model_choices_list()), [], [model_dropdown])

        # ------------------------------------------------------------------ #
        # Call backs: language switch
        # ------------------------------------------------------------------ #
        def switch_language(lang_code: str):
            Tn = LANG_JSON[lang_code]
            # List of all components that need label/value updates
            # This must match the `outputs` of lang_select.change
            # Note: Button text is updated with `value=...`
            # Markdown needs `value=...`
            # Tab labels are updated with `label=...`
            # seed_box_inf was missing from the original outputs list for lang_select.change, adding it.
            return [
                gr.update(label=Tn["language_label"], value=lang_code), # lang_select itself
                # Tab labels
                gr.update(label=Tn["data_process_tab"]), gr.update(label=Tn["train_tab"]), gr.update(label=Tn["infer_tab"]),
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
                # Scheduler specific params also need label updates if they are translated (e.g., "Step Size")
                # Assuming "Step Size", "Step Gamma", "Polynomial Power" are not in LANG_JSON for now.
                # If they are, add them: gr.update(label=Tn["train_step_size"]) etc.
                gr.update(label=Tn["train_save_interval"]),
                gr.update(value=Tn["train_start_btn"]), gr.update(value=Tn["stop_btn"]),
                gr.update(label=Tn["train_log"]), gr.update(label=Tn["train_plot"]),
                # Inference panel
                gr.update(label=Tn["dp_processed_dir"]), gr.update(label=Tn["inf_out_dir"]),
                gr.update(label=Tn["inf_prompt"]), gr.update(label=Tn["inf_num_samples"]),
                gr.update(label=Tn["inf_max_new_tokens"]), gr.update(label=Tn["inf_temperature"]),
                gr.update(label=Tn["inf_top_k"]), gr.update(label=Tn["inf_seed"]), # seed_box_inf label
                gr.update(value=Tn["inf_start_btn"]), gr.update(label=Tn["inf_result"]),
            ]

        lang_select_outputs = [
            lang_select,
            data_process_tab, train_tab, inf_tab,
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
            warmup_box, lr_decay_box, min_lr_box, # step_size, step_gamma, poly_power labels if translated
            save_interval_box, train_btn, stop_btn,
            train_log, train_plot,
            data_dir_inf, out_dir_inf, prompt_box,
            num_samples_box, max_new_tokens_box, temperature_box, top_k_box,
            seed_box_inf, # Added seed_box_inf
            inf_btn, inf_output,
        ]
        # Reconstruct switch_language return to match the full output list for lang_select.change
        # The number of gr.update() calls in switch_language must match len(lang_select_outputs)
        # Original code for switch_language did not update step_size_box etc. labels. Assuming they are not translated.
        # Final items in lang_select_outputs were seed_box_inf, inf_btn, inf_output.
        # switch_language needs to provide updates for all of them.

        # Corrected switch_language to match the outputs list more closely
        # The original switch_language was missing updates for some items in the original lang_select_outputs.
        # For full correctness, ensure every component in lang_select_outputs gets a gr.update in switch_language.
        # The provided switch_language updates many, but a 1-to-1 match is crucial.
        # For brevity, I'll use the user's original switch_language outputs list, assuming it was mostly correct
        # but noting that seed_box_inf was added to outputs and its label update to switch_language.

        lang_select.change(
            fn=switch_language,
            inputs=[lang_select],
            outputs=lang_select_outputs # Use the defined list
        )

        # Initialize LR scheduler params display logic on app load
        demo.load(
            fn=lambda scheduler_type=DEFAULT_CONFIG["training"]["lr_scheduler_type"]: update_lr_scheduler_params(scheduler_type),
            inputs=None, # No explicit inputs, uses default from fn signature
            outputs=[
                warmup_box, lr_decay_box, min_lr_box,
                step_size_box, step_gamma_box, polynomial_power_box
            ]
        )
    return demo

# ----------------- Launch -------------------
if __name__ == "__main__":
    # Ensure LANG_JSON is loaded correctly if build_app_interface is called directly
    # It's loaded from src.config, so should be fine.
    app = build_app_interface() # Default language (e.g. "zh")
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)