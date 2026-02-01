import os
import re
from pathlib import Path

import torch

from src.gpt_model import GPTConfig, GPT
from src.gpt_self_attn import GPTSelfAttnConfig, GPTSelfAttn
from src.sft import chat_generate, tokenize_user_input
from src.ui.html_render import (
    _escape_html,
    _generate_user_tokenization_html,
    _generate_response_html_with_tokens,
    _generate_chat_advanced_html,
)
from src.ui.state import dbm


def chat_cb(user_msg, history, model_sel, sys_prompt, max_tokens, temp, top_k, seed, device_val):
    if not user_msg or not user_msg.strip():
        # Show error for empty message
        import gradio as gr

        gr.Warning("⚠️ Please enter a message.")
        return "", history or [], ""

    # Update history with user message
    history = history or []
    history.append({"role": "user", "content": user_msg})
    yield "", history, ""

    # Validate model
    if not model_sel or " - " not in model_sel:
        history[-1] = {"role": "user", "content": user_msg}
        history.append({"role": "assistant", "content": "❌ Please select a model first."})
        yield "", history, ""
        return

    model_id = int(model_sel.split(" - ")[0])
    model_info = dbm.get_model(model_id)
    if not model_info:
        history[-1] = {"role": "user", "content": user_msg}
        history.append({"role": "assistant", "content": "❌ Model not found."})
        yield "", history, ""
        return

    try:
        ckpt_path = os.path.join(model_info["out_dir"], "ckpt.pt")
        sft_ckpt_path = os.path.join(model_info["out_dir"], "sft", "ckpt_sft.pt")

        # Prefer SFT checkpoint if available
        load_path = sft_ckpt_path if os.path.exists(sft_ckpt_path) else ckpt_path

        # Check for tokenizer
        tokenizer_path = Path.cwd() / "assets" / "tokenizer.json"
        if not tokenizer_path.exists():
            history[-1] = {"role": "user", "content": user_msg}
            history.append({"role": "assistant", "content": "❌ tokenizer.json not found."})
            yield "", history, "", ""
            return

        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(tokenizer_path))

        # Load meta.pkl to get old2new mapping for token ID remapping
        processed_data_dir = model_info.get("processed_data_dir", "")
        meta_path = os.path.join(processed_data_dir, "meta.pkl") if processed_data_dir else None
        old2new_mapping = None
        new2old_mapping = None

        if meta_path and os.path.exists(meta_path):
            import pickle

            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            old2new_mapping = meta.get("old2new_mapping") or meta.get("old2new")
            if old2new_mapping:
                new2old_mapping = {new_id: old_id for old_id, new_id in old2new_mapping.items()}

        if old2new_mapping is None:
            history[-1] = {"role": "user", "content": user_msg}
            history.append({"role": "assistant", "content": f"❌ Error: Token ID mapping not found. meta_path={meta_path}, exists={os.path.exists(meta_path) if meta_path else False}. Please ensure the model was trained with the custom tokenizer."})
            yield "", history, "", ""
            return

        # Load saved token IDs from database (if available) to avoid re-tokenization issues
        chat_history_data = dbm.get_chat_history(model_id)
        saved_token_ids = chat_history_data.get("token_ids", [])
        saved_system_prompt = chat_history_data.get("system_prompt", "")

        # Build history_token_ids from saved data
        # IMPORTANT: If system prompt has changed, we must NOT use cached token IDs
        # because the cached IDs contain the old system prompt at the beginning
        history_token_ids = None
        if saved_token_ids:
            # Check if system prompt has changed
            # Compare saved system prompt with current one
            # If saved_system_prompt is empty but we have saved_token_ids, treat as mismatch
            # (this handles edge cases where system_prompt wasn't saved properly)
            prompts_match = (saved_system_prompt == sys_prompt) if saved_system_prompt else (not sys_prompt)
            if not prompts_match:
                # System prompt changed - clear history_token_ids to force re-tokenization
                # This ensures the new system prompt is used
                print(f"⚠️ System prompt changed ('{saved_system_prompt[:30]}...' -> '{sys_prompt[:30] if sys_prompt else ''}...'), clearing token ID cache")
                saved_token_ids = []  # Clear saved token IDs since they contain old system prompt
            elif "all_token_ids" in saved_token_ids[-1]:
                history_token_ids = saved_token_ids[-1]["all_token_ids"]

        # Tokenize user input for display
        user_tokens_info = tokenize_user_input(tokenizer, user_msg, old2new_mapping)
        user_tokenization_html = _generate_user_tokenization_html(user_tokens_info)

        # Tokenize system prompt for display (only for first message or when changed)
        system_prompt_tokens = tokenize_user_input(tokenizer, sys_prompt, old2new_mapping) if sys_prompt else []

        # Load model
        checkpoint = torch.load(load_path, map_location=device_val)
        model_args = checkpoint["model_args"]

        # Determine model type
        is_self_attention_model = any(
            key in model_args for key in ["ffn_hidden_mult", "qkv_bias", "attn_dropout", "resid_dropout"]
        )

        if is_self_attention_model:
            gptconf = GPTSelfAttnConfig(**model_args)
            model = GPTSelfAttn(gptconf)
        else:
            gptconf = GPTConfig(**model_args)
            model = GPT(gptconf)

        state_dict = checkpoint["model"]
        unwanted_prefix = "_orig_mod."
        for k in list(state_dict.keys()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        model.to(device_val)
        model.eval()

        # Prepare messages from history
        messages = []
        for msg in history[:-1]:
            # Handle both old tuple format and new messages format
            if isinstance(msg, dict):
                content = msg.get("content", "")
                role = msg.get("role", "user")
                if isinstance(content, str):
                    plain_content = re.sub(r"<[^>]+>", "", content)
                    plain_content = re.sub(r"📝 Tokens:.*", "", plain_content, flags=re.DOTALL).strip()
                else:
                    plain_content = str(content)
                messages.append({"role": role, "content": plain_content})
            else:
                # Legacy tuple format (user_entry, bot_entry)
                user_entry, bot_entry = msg
                if isinstance(user_entry, str):
                    plain_user = re.sub(r"<[^>]+>", "", user_entry)
                    plain_user = re.sub(r"📝 Tokens:.*", "", plain_user, flags=re.DOTALL).strip()
                else:
                    plain_user = str(user_entry)
                messages.append({"role": "user", "content": plain_user})
                if bot_entry:
                    if isinstance(bot_entry, str):
                        plain_bot = re.sub(r"<[^>]+>", "", bot_entry)
                    else:
                        plain_bot = str(bot_entry)
                    messages.append({"role": "assistant", "content": plain_bot})

        messages.append({"role": "user", "content": user_msg})

        # Set random seed for reproducibility
        torch.manual_seed(int(seed))
        if "cuda" in device_val:
            torch.cuda.manual_seed(int(seed))

        # Generate with proper token ID mappings and detailed info
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
            return_detailed_info=True,
            history_token_ids=history_token_ids,
        )

        all_token_details = []
        response_tokens = []
        final_token_data = None

        # Create user message HTML with tokenization
        user_msg_html = f'<div style="font-family: system-ui, sans-serif;">{_escape_html(user_msg)}{user_tokenization_html}</div>'

        for item in generator:
            text_piece, token_detail = item

            # Check if this is the final message with token IDs
            if token_detail and token_detail.get("is_final"):
                final_token_data = token_detail
                continue

            # Skip empty text pieces
            if not text_piece:
                continue

            # Collect token info
            response_tokens.append({"text": text_piece, "token_detail": token_detail})
            if token_detail:
                all_token_details.append(token_detail)

            # Generate response HTML with token highlighting
            response_html = f'<div style="font-family: system-ui, sans-serif;">{_generate_response_html_with_tokens(response_tokens)}</div>'

            # Update history with user and assistant messages
            # On first token: history[-1] is user message, need to add assistant message
            # On subsequent tokens: history[-1] is assistant message, need to update it
            if history[-1].get("role") == "user":
                # First token: update user message with HTML and add assistant message
                history[-1] = {"role": "user", "content": user_msg_html}
                history.append({"role": "assistant", "content": response_html})
            else:
                # Subsequent tokens: just update assistant message
                history[-1] = {"role": "assistant", "content": response_html}

            # Generate advanced HTML (include system prompt tokens only for first turn)
            show_sys_tokens = system_prompt_tokens if len(history) == 2 else None
            advanced_html = _generate_chat_advanced_html(all_token_details, response_tokens, show_sys_tokens)

            yield "", history, advanced_html

        # Final update
        response_html = f'<div style="font-family: system-ui, sans-serif;">{_generate_response_html_with_tokens(response_tokens)}</div>'
        # Ensure the last two messages are user and assistant
        if len(history) >= 2 and history[-2].get("role") == "user":
            history[-2] = {"role": "user", "content": user_msg_html}
            history[-1] = {"role": "assistant", "content": response_html}
        else:
            # Fallback: replace last with user, append assistant
            history[-1] = {"role": "user", "content": user_msg_html}
            history.append({"role": "assistant", "content": response_html})
        show_sys_tokens = system_prompt_tokens if len(history) == 2 else None
        advanced_html = _generate_chat_advanced_html(all_token_details, response_tokens, show_sys_tokens)

        # Update saved token IDs with the new conversation turn
        if final_token_data:
            saved_token_ids.append(
                {
                    "all_token_ids": final_token_data.get("all_token_ids", []),
                    "generated_token_ids": final_token_data.get("generated_token_ids", []),
                    "prompt_length": final_token_data.get("prompt_length", 0),
                }
            )

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
        # Update history with error message
        if len(history) > 0 and history[-1].get("role") == "user":
            history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        else:
            history[-1] = {"role": "assistant", "content": f"❌ Error: {str(e)}"}
        yield "", history, f'<div style="color: red;">Error: {_escape_html(str(e))}</div>'


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
