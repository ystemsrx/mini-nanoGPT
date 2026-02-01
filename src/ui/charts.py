import numpy as np


def generate_loss_chart_html(
    train_data,  # List of (epoch, loss) tuples
    val_data,    # List of (epoch, loss) tuples
):
    svg_width = 800
    svg_height = 400
    margin_left = 60  # For Y-axis labels
    margin_top = 50
    margin_bottom = 60  # For X-axis title
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
        data_min_loss = min(all_losses) if all_losses else 0.0  # Ensure min isn't called on empty
        data_max_loss = max(all_losses) if all_losses else 1.0  # Ensure max isn't called on empty

    loss_range = data_max_loss - data_min_loss
    if loss_range < 0.01:  # Handle very flat data
        y_axis_min_display = max(0.0, data_min_loss - 0.05)
        y_axis_max_display = y_axis_min_display + 0.1
    else:
        y_axis_min_display = max(0.0, data_min_loss - loss_range * 0.15)
        y_axis_max_display = data_max_loss + loss_range * 0.15

    if y_axis_min_display >= y_axis_max_display:  # Ensure max > min
        y_axis_max_display = y_axis_min_display + 0.1

    def to_svg_coords(epoch, loss):
        if display_max_epoch == display_min_epoch:  # Avoid division by zero
            x_scaled = 0
        else:
            x_scaled = (epoch - display_min_epoch) / (display_max_epoch - display_min_epoch)
        x = margin_left + x_scaled * chart_width

        y_display_range = y_axis_max_display - y_axis_min_display
        if y_display_range == 0:  # Avoid division by zero
            y_scaled = 0
        else:
            y_scaled = (loss - y_axis_min_display) / y_display_range

        y = margin_top + chart_height - y_scaled * chart_height  # Invert Y for SVG

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

            _catmull_rom_tension = 1 / 6

            if len(svg_points) == 1:
                p0 = svg_points[0]
                path_static_d = f"M {p0[0]},{p0[1]}"
                area_d = f"M {p0[0]},{p0[1]} L {p0[0]},{axis_base_y} L {p0[0]},{axis_base_y} Z"

            elif len(svg_points) >= 2:
                control_points_list = []
                for i in range(len(svg_points) - 1):
                    p_i = svg_points[i]
                    p_i_plus_1 = svg_points[i + 1]

                    p_i_minus_1 = svg_points[i - 1] if i > 0 else p_i
                    k1_x = p_i[0] + (p_i_plus_1[0] - p_i_minus_1[0]) * _catmull_rom_tension
                    k1_y = p_i[1] + (p_i_plus_1[1] - p_i_minus_1[1]) * _catmull_rom_tension

                    p_i_plus_2 = svg_points[i + 2] if i + 2 < len(svg_points) else p_i_plus_1
                    k2_x = p_i_plus_1[0] - (p_i_plus_2[0] - p_i[0]) * _catmull_rom_tension
                    k2_y = p_i_plus_1[1] - (p_i_plus_2[1] - p_i[1]) * _catmull_rom_tension
                    control_points_list.append((k1_x, k1_y, k2_x, k2_y))

                full_smooth_path_for_area = f"M {svg_points[0][0]},{svg_points[0][1]}"
                for i in range(len(svg_points) - 1):
                    k1x, k1y, k2x, k2y = control_points_list[i]
                    p_next = svg_points[i + 1]
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
                        p_next = svg_points[i + 1]
                        path_static_d += f" C {round(k1x,2)},{round(k1y,2)} {round(k2x,2)},{round(k2y,2)} {p_next[0]},{p_next[1]}"

                    k1x_anim, k1y_anim, k2x_anim, k2y_anim = control_points_list[-1]
                    p_prev_anim = svg_points[-2]
                    p_curr_anim = svg_points[-1]
                    path_anim_d = f"M {p_prev_anim[0]},{p_prev_anim[1]} C {round(k1x_anim,2)},{round(k1y_anim,2)} {round(k2x_anim,2)},{round(k2y_anim,2)} {p_curr_anim[0]},{p_curr_anim[1]}"

                if len(svg_points) >= 2:  # Ensure there are at least two points for segment length
                    p_prev_for_len = svg_points[-2]
                    p_curr_for_len = svg_points[-1]
                    anim_segment_length = np.sqrt((p_curr_for_len[0] - p_prev_for_len[0]) ** 2 + (p_curr_for_len[1] - p_prev_for_len[1]) ** 2)
                    anim_segment_length = max(0.1, round(anim_segment_length, 2))
                else:  # Should not happen if len(svg_points) >= 2, but as a fallback
                    anim_segment_length = 0.1

        return path_static_d, path_anim_d, circles_svg, area_d, anim_segment_length

    train_path_static_d, train_path_anim_d, train_circles_svg, train_area_d, train_anim_segment_length = create_path_elements(train_data, "train")
    val_path_static_d, val_path_anim_d, val_circles_svg, val_area_d, val_anim_segment_length = create_path_elements(val_data, "val")

    x_axis_labels_svg = ""
    num_x_ticks = 5
    effective_display_max_epoch = max(display_min_epoch, display_max_epoch)
    x_tick_values = np.linspace(display_min_epoch, effective_display_max_epoch, num_x_ticks + 1)
    if effective_display_max_epoch == display_min_epoch and effective_display_max_epoch == 0:  # Handles case of single point at 0 or no data
        x_tick_values = np.linspace(0, 10, num_x_ticks + 1)  # Default axis if no data
    elif effective_display_max_epoch == display_min_epoch:  # Single point not at 0
        x_tick_values = [display_min_epoch]

    for epoch_val in x_tick_values:
        x_coord, _ = to_svg_coords(epoch_val, y_axis_min_display)
        label = f"{epoch_val:.1f}" if effective_display_max_epoch < 10 and effective_display_max_epoch != 0 else f"{int(round(epoch_val))}"
        x_axis_labels_svg += f'<text class="axis-label" x="{x_coord}" y="{axis_base_y + 25}">{label}</text>\n'
    x_axis_labels_svg += f'<text class="axis-title" x="{margin_left + chart_width / 2}" y="{axis_base_y + 45}">Steps</text>\n'  # Changed from Epoch to Steps

    y_axis_labels_svg = ""
    num_y_ticks = 5
    y_tick_values = np.linspace(y_axis_min_display, y_axis_max_display, num_y_ticks + 1)
    if y_axis_min_display == y_axis_max_display:
        y_tick_values = [y_axis_min_display] if y_axis_min_display != 0 else np.linspace(0, 1, num_y_ticks + 1)

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
            main_y_end = y_tick_values[i + 1]
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


def make_progress_html(progress_val, max_val, color="blue"):
    """Generate HTML for a progress bar."""
    return (
        f"<div style='width: 100%; height: 20px; margin-bottom: 5px;'>"
        f"<progress value='{progress_val}' max='{max_val if max_val > 0 else 1}' "
        f"style='width: 100%; height: 20px; color: {color};'></progress>"
        "</div>"
    )
