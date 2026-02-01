from .constants import TOKEN_COLORS


def _escape_html(text):
    """Escape HTML special characters"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
        .replace("\n", "<br>")
    )


def _generate_token_html(tokens_with_text, prompt_text="", prompt_tokens=None):
    """
    Generate HTML with token highlighting
    tokens_with_text: list of dicts with 'text' and optionally 'token_detail'
    prompt_text: raw prompt text (used if prompt_tokens is None)
    prompt_tokens: list of dicts with 'text', 'original_id', 'mapped_id', 'in_vocab' for each prompt token
    """
    html_parts = []
    html_parts.append('<div style="font-family: monospace; white-space: pre-wrap; line-height: 1.6; padding: 10px; background: #ffffff; border-radius: 8px; border: 1px solid #e0e0e0;">')

    # Add prompt part with RED border box to highlight user input
    if prompt_tokens:
        html_parts.append('<span style="display: inline; border: 2px solid #e53935; border-radius: 4px; padding: 2px 4px; background: rgba(229, 57, 53, 0.08); margin-right: 4px;">')
        for i, token_info in enumerate(prompt_tokens):
            text = token_info.get("text", "")
            if not text:
                continue
            escaped_text = _escape_html(text)
            color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
            orig_id = token_info.get("original_id", "?")
            mapped_id = token_info.get("mapped_id", "?")
            in_vocab = token_info.get("in_vocab", True)
            border_color = "#4caf50" if in_vocab else "#f44336"
            tooltip = f"Prompt Token #{i+1}&#10;Text: '{escaped_text}'&#10;Original ID: {orig_id}&#10;Mapped ID: {mapped_id}&#10;In Vocab: {'Yes' if in_vocab else 'No'}"
            html_parts.append(f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 3px; border-bottom: 2px solid {border_color}; cursor: help;" title="{tooltip}">{escaped_text}</span>')
        html_parts.append("</span>")
    elif prompt_text:
        # Fallback: RED border box for prompt without token details
        escaped_prompt = _escape_html(prompt_text)
        html_parts.append(f'<span style="display: inline; border: 2px solid #e53935; border-radius: 4px; padding: 2px 4px; background: rgba(229, 57, 53, 0.08); color: #333;">{escaped_prompt}</span>')

    # Add generated tokens with highlighting
    color_idx = 0
    for item in tokens_with_text:
        text = item.get("text", "")
        if not text:
            continue

        escaped_text = _escape_html(text)
        color = TOKEN_COLORS[color_idx % len(TOKEN_COLORS)]

        # Create highlighted span with tooltip
        token_detail = item.get("token_detail")
        if token_detail:
            candidates = token_detail.get("top5_candidates", [])
            selected_id = token_detail.get("selected_token_id")
            tooltip_parts = [f"#{token_detail.get('position', '?')}: {escaped_text}"]
            for i, cand in enumerate(candidates[:6]):  # Show up to 6 (selected + top 5)
                prob_pct = cand["probability"] * 100
                cand_text = _escape_html(cand["text"])
                # Check if this candidate is the selected one (using is_selected field or token_id comparison)
                is_selected = cand.get("is_selected", False) or cand["token_id"] == selected_id
                marker = "→" if is_selected else " "
                tooltip_parts.append(f"{marker}{i+1}. '{cand_text}' ({prob_pct:.1f}%)")
            tooltip = "&#10;".join(tooltip_parts)
            html_parts.append(f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 3px; cursor: help;" title="{tooltip}">{escaped_text}</span>')
        else:
            html_parts.append(f'<span style="background-color: {color}; padding: 1px 2px; border-radius: 3px;">{escaped_text}</span>')

        color_idx += 1

    html_parts.append("</div>")
    return "".join(html_parts)


def _generate_advanced_html(all_token_details):
    """
    Generate detailed HTML for the advanced output panel with fresh and elegant styling
    """
    html_parts = []
    html_parts.append('<div style="font-family: system-ui, -apple-system, sans-serif; font-size: 13px;">')

    for sample_info in all_token_details:
        sample_idx = sample_info.get("sample_index", 0)
        token_details = sample_info.get("token_details", [])

        # Sample header style
        html_parts.append(f'<div style="margin: 16px 0 10px 0; padding: 6px 16px; background: #e0f7fa; color: #006064; border-radius: 20px; font-weight: 600; display: inline-block; font-size: 13px; border: 1px solid #b2ebf2;">Sample {sample_idx + 1}</div>')

        # Table with inline styles
        html_parts.append('<div style="max-height: 500px; overflow-y: auto; margin-bottom: 24px; border: 1px solid #f0f0f0; border-radius: 8px;">')
        html_parts.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px; background: white;">')

        # Header style
        html_parts.append('<thead><tr style="background: #f8f9fa; border-bottom: 1px solid #e9ecef;">')
        html_parts.append('<th style="padding: 12px 16px; text-align: center; color: #5f6368; font-weight: 600; width: 60px;">#</th>')
        html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600; width: 140px;">Selected</th>')
        html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600;">Top 5 Candidates</th>')
        html_parts.append('</tr></thead><tbody>')

        for row_idx, detail in enumerate(token_details):
            pos = detail.get("position", 0)
            selected_text = _escape_html(detail.get("selected_token_text", ""))
            selected_id = detail.get("selected_token_id", -1)
            candidates = detail.get("top5_candidates", [])

            # Find selected probability
            selected_prob = 0
            non_selected_candidates = []
            for cand in candidates:
                is_selected = cand.get("is_selected", False) or cand["token_id"] == selected_id
                if is_selected:
                    selected_prob = cand["probability"] * 100
                else:
                    non_selected_candidates.append(cand)

            # Only show first 5 non-selected candidates with capsule style
            cand_html_parts = []
            for idx, cand in enumerate(non_selected_candidates[:5]):
                prob_pct = cand["probability"] * 100
                cand_text = _escape_html(cand["text"])
                # Capsule style with shadow
                cand_html_parts.append(
                    f'<span style="display: inline-block; margin: 3px 6px 3px 0; padding: 4px 10px; '
                    f'background: #ffffff; color: #555; '
                    f'border-radius: 12px; font-size: 12px; white-space: nowrap; '
                    f'box-shadow: 0 1px 2px rgba(0,0,0,0.08); border: 1px solid #ebebeb;">'
                    f'<span style="color: #bbb; font-size: 10px; margin-right: 4px;">#{idx+1}</span>'
                    f'{cand_text} <span style="color: #999; font-size: 11px;">({prob_pct:.1f}%)</span></span>'
                )

            candidates_html = "".join(cand_html_parts) if cand_html_parts else '<span style="color: #ccc;">-</span>'

            # Row background alternation
            row_bg = "#fbfbfb" if row_idx % 2 == 1 else "white"

            html_parts.append(f'<tr style="background: {row_bg}; border-bottom: 1px solid #f5f5f5;">')
            html_parts.append(f'<td style="padding: 10px 16px; text-align: center; color: #9aa0a6; font-size: 12px;">{pos + 1}</td>')

            # Selected cell style
            html_parts.append(f'<td style="padding: 10px 16px;"><span style="background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 6px; padding: 4px 8px; display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; font-weight: 500;">{selected_text} <span style="opacity: 0.7; font-size: 11px; margin-left: 2px;">{selected_prob:.1f}%</span></span></td>')
            html_parts.append(f'<td style="padding: 10px 16px; line-height: 1.6;">{candidates_html}</td>')
            html_parts.append("</tr>")

        html_parts.append("</tbody></table></div>")

    html_parts.append("</div>")
    return "".join(html_parts)


def _generate_user_tokenization_html(tokens_info):
    """Generate HTML to display user input tokenization - inline with message"""
    if not tokens_info:
        return ""

    html_parts = []
    html_parts.append('<div style="display: flex; flex-wrap: wrap; gap: 3px; margin-top: 6px; padding-top: 6px; border-top: 1px dashed #ccc;">')
    html_parts.append('<span style="font-size: 11px; color: #666; margin-right: 4px;">📝 Tokens:</span>')

    for i, token_info in enumerate(tokens_info):
        text = _escape_html(token_info["text"])
        orig_id = token_info["original_id"]
        mapped_id = token_info["mapped_id"]
        in_vocab = token_info["in_vocab"]

        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        border_color = "#4caf50" if in_vocab else "#f44336"
        border_style = "1px solid " + border_color

        tooltip = f"Token #{i+1}&#10;Text: '{text}'&#10;Original ID: {orig_id}&#10;Mapped ID: {mapped_id}&#10;In Vocab: {'Yes' if in_vocab else 'No'}"

        html_parts.append(
            f'''<span style="background-color: {color}; padding: 1px 4px; border-radius: 3px; 
                    border: {border_style}; cursor: help; font-size: 11px;" title="{tooltip}">{text}</span>'''
        )

    html_parts.append("</div>")
    return "".join(html_parts)


def _generate_response_html_with_tokens(response_tokens):
    """Generate HTML for response text with token highlighting"""
    if not response_tokens:
        return ""

    html_parts = []
    for i, token_info in enumerate(response_tokens):
        text = _escape_html(token_info["text"])
        color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
        detail = token_info.get("token_detail")

        if detail:
            candidates = detail.get("top5_candidates", [])
            tooltip_parts = [f"#{detail.get('position', i)+1}: '{text}'"]
            for j, cand in enumerate(candidates[:5]):
                prob_pct = cand["probability"] * 100
                cand_text = _escape_html(cand["text"])
                marker = "→" if cand["token_id"] == detail.get("selected_token_id") else " "
                tooltip_parts.append(f"{marker}{j+1}. '{cand_text}' ({prob_pct:.1f}%)")
            tooltip = "&#10;".join(tooltip_parts)
        else:
            tooltip = f"Token: {text}"

        html_parts.append(
            f'''<span style="background-color: {color}; padding: 0px 2px; border-radius: 2px; 
                    cursor: help;" title="{tooltip}">{text}</span>'''
        )

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
            text = _escape_html(token_info["text"])
            orig_id = token_info["original_id"]
            mapped_id = token_info["mapped_id"]
            in_vocab = token_info["in_vocab"]
            color = TOKEN_COLORS[i % len(TOKEN_COLORS)]
            border_color = "#4caf50" if in_vocab else "#f44336"
            tooltip = f"Token #{i+1}&#10;Text: '{text}'&#10;Original ID: {orig_id}&#10;Mapped ID: {mapped_id}&#10;In Vocab: {'Yes' if in_vocab else 'No'}"
            html_parts.append(
                f'''<span style="background-color: {color}; padding: 2px 6px; border-radius: 4px; border: 1px solid {border_color}; cursor: help; font-size: 12px;" title="{tooltip}">{text}</span>'''
            )
        html_parts.append("</div>")

    if not all_token_details:
        html_parts.append("</div>")
        return "".join(html_parts)

    # Response header style
    html_parts.append('<div style="margin: 16px 0 10px 0; padding: 6px 16px; background: #e0f7fa; color: #006064; border-radius: 20px; font-weight: 600; display: inline-block; font-size: 13px; border: 1px solid #b2ebf2;">Response Token Details</div>')

    # Table with inline styles (matching non-chat mode)
    html_parts.append('<div style="max-height: 500px; overflow-y: auto; margin-bottom: 24px; border: 1px solid #f0f0f0; border-radius: 8px;">')
    html_parts.append('<table style="width: 100%; border-collapse: collapse; font-size: 13px; background: white;">')

    # Header style
    html_parts.append('<thead><tr style="background: #f8f9fa; border-bottom: 1px solid #e9ecef;">')
    html_parts.append('<th style="padding: 12px 16px; text-align: center; color: #5f6368; font-weight: 600; width: 60px;">#</th>')
    html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600; width: 140px;">Selected</th>')
    html_parts.append('<th style="padding: 12px 16px; text-align: left; color: #5f6368; font-weight: 600;">Top 5 Candidates</th>')
    html_parts.append('</tr></thead><tbody>')

    for row_idx, detail in enumerate(all_token_details):
        pos = detail.get("position", 0)
        selected_text = _escape_html(detail.get("selected_token_text", ""))
        selected_id = detail.get("selected_token_id", -1)
        candidates = detail.get("top5_candidates", [])

        # Find selected probability
        selected_prob = 0
        non_selected_candidates = []
        for cand in candidates:
            is_selected = cand.get("is_selected", False) or cand["token_id"] == selected_id
            if is_selected:
                selected_prob = cand["probability"] * 100
            else:
                non_selected_candidates.append(cand)

        # Only show first 5 non-selected candidates with capsule style
        cand_html_parts = []
        for idx, cand in enumerate(non_selected_candidates[:5]):
            prob_pct = cand["probability"] * 100
            cand_text = _escape_html(cand["text"])
            # Capsule style with shadow
            cand_html_parts.append(
                f'<span style="display: inline-block; margin: 3px 6px 3px 0; padding: 4px 10px; '
                f'background: #ffffff; color: #555; '
                f'border-radius: 12px; font-size: 12px; white-space: nowrap; '
                f'box-shadow: 0 1px 2px rgba(0,0,0,0.08); border: 1px solid #ebebeb;">'
                f'<span style="color: #bbb; font-size: 10px; margin-right: 4px;">#{idx+1}</span>'
                f'{cand_text} <span style="color: #999; font-size: 11px;">({prob_pct:.1f}%)</span></span>'
            )

        candidates_html = "".join(cand_html_parts) if cand_html_parts else '<span style="color: #ccc;">-</span>'

        # Row background alternation
        row_bg = "#fbfbfb" if row_idx % 2 == 1 else "white"

        html_parts.append(f'<tr style="background: {row_bg}; border-bottom: 1px solid #f5f5f5;">')
        html_parts.append(f'<td style="padding: 10px 16px; text-align: center; color: #9aa0a6; font-size: 12px;">{pos + 1}</td>')

        # Selected cell style
        html_parts.append(
            f'<td style="padding: 10px 16px;"><span style="background: #e8f5e9; color: #2e7d32; border: 1px solid #c8e6c9; border-radius: 6px; padding: 4px 8px; display: inline-block; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; font-weight: 500;">{selected_text} <span style="opacity: 0.7; font-size: 11px; margin-left: 2px;">{selected_prob:.1f}%</span></span></td>'
        )
        html_parts.append(f'<td style="padding: 10px 16px; line-height: 1.6;">{candidates_html}</td>')
        html_parts.append("</tr>")

    html_parts.append("</tbody></table></div>")
    html_parts.append("</div>")
    return "".join(html_parts)
