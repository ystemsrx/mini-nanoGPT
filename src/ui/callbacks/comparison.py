import concurrent.futures
import os
import pickle
import queue
import re
import time
from pathlib import Path

import gradio as gr
import torch

from src.config import DEFAULT_CONFIG
from src.device_manager import device_manager
from src.gpt_model import GPTConfig, GPT
from src.gpt_self_attn import GPTSelfAttnConfig, GPTSelfAttn
from src.infer_cache import cached_generate_text, ModelCache, UnknownTokenError, stop_inference
from src.sft import chat_generate, tokenize_user_input, stop_chat_generation
from src.ui.charts import generate_loss_chart_html
from src.ui.helpers import _create_plot_html_from_log
from src.ui.html_render import (
    _escape_html,
    _generate_token_html,
    _generate_user_tokenization_html,
    _generate_response_html_with_tokens,
)
from src.ui.state import dbm


def _render_error_html(message: str) -> str:
    return (
        "<div style='color: red; padding: 10px; background: #ffe6e6; border-radius: 8px;'>"
        f"{_escape_html(message)}"
        "</div>"
    )


def _load_sft_loss_plot_html(model_id: int) -> str:
    sft_loss_log_path = dbm.get_sft_log_path(model_id)
    if not (sft_loss_log_path and os.path.exists(sft_loss_log_path)):
        return ""
    try:
        with open(sft_loss_log_path, "rb") as f:
            sft_log_data = pickle.load(f)
        sft_tr_steps = sft_log_data.get("train_steps", [])
        sft_tr_losses = sft_log_data.get("train_losses", [])
        if sft_tr_steps and sft_tr_losses:
            train_data = list(zip(sft_tr_steps, sft_tr_losses))
            return generate_loss_chart_html(train_data, [])
    except Exception as e:
        print(f"Error loading SFT loss log: {e}")
    return ""


def _combine_loss_plots(base_html: str, sft_html: str) -> str:
    return (
        '<div style="display: flex; gap: 12px; flex-wrap: wrap;">'
        '<div style="flex: 1; min-width: 260px;">'
        '<div style="font-weight: 600; margin-bottom: 6px;">Pretrain</div>'
        f"{base_html}"
        "</div>"
        '<div style="flex: 1; min-width: 260px;">'
        '<div style="font-weight: 600; margin-bottom: 6px;">SFT</div>'
        f"{sft_html}"
        "</div>"
        "</div>"
    )


def _tokenize_prompt_for_display(data_dir: str, prompt: str):
    prompt_tokens = None
    if not data_dir:
        return None
    try:
        meta_path = os.path.join(data_dir, "meta.pkl")
        if os.path.exists(meta_path):
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            old2new_mapping = meta.get("old2new_mapping") or meta.get("old2new")
            tokenizer_type = meta.get("tokenizer_type") or meta.get("tokenizer", "char_level")

            if tokenizer_type == "custom_json" and old2new_mapping:
                tokenizer_path = Path.cwd() / "assets" / "tokenizer.json"
                if tokenizer_path.exists():
                    from tokenizers import Tokenizer

                    tokenizer = Tokenizer.from_file(str(tokenizer_path))
                    prompt_tokens = tokenize_user_input(tokenizer, prompt, old2new_mapping)
            elif tokenizer_type == "gpt2" and old2new_mapping:
                import tiktoken

                enc = tiktoken.get_encoding("gpt2")
                ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
                prompt_tokens = []
                for orig_id in ids:
                    try:
                        decoded_text = enc.decode([orig_id])
                    except Exception:
                        decoded_text = f"<token_{orig_id}>"
                    mapped_id = old2new_mapping.get(orig_id, orig_id)
                    in_vocab = orig_id in old2new_mapping
                    prompt_tokens.append(
                        {
                            "text": decoded_text,
                            "original_id": orig_id,
                            "mapped_id": mapped_id,
                            "in_vocab": in_vocab,
                        }
                    )
            else:
                stoi = meta.get("stoi", {})
                prompt_tokens = []
                for ch in prompt:
                    in_vocab = ch in stoi
                    prompt_tokens.append(
                        {
                            "text": ch,
                            "original_id": stoi.get(ch, -1),
                            "mapped_id": stoi.get(ch, -1),
                            "in_vocab": in_vocab,
                        }
                    )
    except Exception as e:
        print(f"Warning: Failed to tokenize prompt for display: {e}")
        prompt_tokens = None
    return prompt_tokens


def _init_stream_state(prompt: str, prompt_tokens):
    return {
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "current_sample_tokens": [],
        "all_samples_info": [],
        "current_sample_idx": 0,
        "prompt_displayed": False,
    }


def _update_stream_state(state, text_piece, token_detail):
    if not text_piece:
        return
    if text_piece.startswith("Sample ") and text_piece.endswith(":\n"):
        if state["current_sample_tokens"]:
            state["all_samples_info"].append(
                {
                    "sample_idx": state["current_sample_idx"] - 1,
                    "prompt": state["prompt"],
                    "tokens": state["current_sample_tokens"].copy(),
                }
            )
        state["current_sample_tokens"] = []
        state["prompt_displayed"] = False
        try:
            state["current_sample_idx"] = int(text_piece.replace("Sample ", "").replace(":\n", ""))
        except ValueError:
            state["current_sample_idx"] = max(state["current_sample_idx"], 1)
        return
    if text_piece == state["prompt"] and not state["prompt_displayed"]:
        state["prompt_displayed"] = True
        return
    if text_piece.startswith("\n" + "-" * 30):
        if state["current_sample_tokens"]:
            state["all_samples_info"].append(
                {
                    "sample_idx": state["current_sample_idx"] - 1,
                    "prompt": state["prompt"],
                    "tokens": state["current_sample_tokens"].copy(),
                }
            )
        state["current_sample_tokens"] = []
        return

    state["current_sample_tokens"].append({"text": text_piece, "token_detail": token_detail})


def _finalize_stream_state(state):
    if state["current_sample_tokens"]:
        state["all_samples_info"].append(
            {
                "sample_idx": state["current_sample_idx"] - 1,
                "prompt": state["prompt"],
                "tokens": state["current_sample_tokens"].copy(),
            }
        )
        state["current_sample_tokens"] = []
    state["prompt_displayed"] = False
    state["current_sample_idx"] = 0


def _render_stream_html(state) -> str:
    html_parts = []
    html_parts.append(
        '<div style="font-family: system-ui, sans-serif; background: #ffffff; border: 2px solid #000000; '
        'border-radius: 8px; padding: 15px;">'
    )

    for sample_info in state["all_samples_info"]:
        html_parts.append(
            f'<div style="margin-bottom: 15px;"><strong>Sample {sample_info["sample_idx"] + 1}:</strong><br>'
        )
        html_parts.append(
            _generate_token_html(
                sample_info["tokens"],
                prompt_text=state["prompt"],
                prompt_tokens=state["prompt_tokens"],
            )
        )
        html_parts.append("</div>")

    if state["current_sample_tokens"] or state["prompt_displayed"]:
        html_parts.append(
            f'<div style="margin-bottom: 15px;"><strong>Sample {state["current_sample_idx"]}:</strong><br>'
        )
        prompt_tokens = state["prompt_tokens"] if state["prompt_displayed"] else None
        prompt_text = state["prompt"] if state["prompt_displayed"] else ""
        html_parts.append(
            _generate_token_html(
                state["current_sample_tokens"],
                prompt_text=prompt_text,
                prompt_tokens=prompt_tokens,
            )
        )
        html_parts.append("</div>")

    html_parts.append("</div>")
    return "".join(html_parts)


def _sanitize_chat_content(content):
    if not isinstance(content, str):
        return str(content)
    plain_content = re.sub(r"<[^>]+>", "", content)
    plain_content = re.sub(r"📝 Tokens:.*", "", plain_content, flags=re.DOTALL).strip()
    return plain_content


def _build_messages_from_history(history):
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            role = msg.get("role", "user")
            content = _sanitize_chat_content(msg.get("content", ""))
            messages.append({"role": role, "content": content})
        else:
            user_entry, bot_entry = msg
            messages.append({"role": "user", "content": _sanitize_chat_content(user_entry)})
            if bot_entry:
                messages.append({"role": "assistant", "content": _sanitize_chat_content(bot_entry)})
    return messages


def _load_old2new_mapping(processed_data_dir: str):
    if not processed_data_dir:
        return None
    meta_path = os.path.join(processed_data_dir, "meta.pkl")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    return meta.get("old2new_mapping") or meta.get("old2new")


def select_model_for_comparison_cb(sel: str, is_left: bool):
    """
    Select model for comparison (left or right side)
    """
    if not sel:
        return [{}, generate_loss_chart_html([], []), "", "", "", False]

    try:
        mid = int(sel.split(" - ")[0])
    except ValueError:
        return [{}, generate_loss_chart_html([], []), "", "", "", False]

    cfg = dbm.get_training_config(mid) or {}
    sft_cfg = dbm.get_sft_config(mid) or {}
    info = dbm.get_model_basic_info(mid) or {}
    name = info.get("name", "unknown_model")

    if "dir_path" in info:
        out_dir_root = info["dir_path"]
        folder = os.path.basename(out_dir_root)
        data_processed_dir = os.path.join("data", folder, "processed")
    else:
        folder_name_part = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in name)
        folder = f"{folder_name_part}_{mid}"
        data_processed_dir = os.path.join("data", folder, "processed")
        out_dir_root = os.path.join("out", folder)

    has_sft_ckpt = os.path.exists(os.path.join(out_dir_root, "sft", "ckpt_sft.pt"))

    loss_log_path = dbm.get_training_log_path(mid)
    base_loss_plot_html = _create_plot_html_from_log(loss_log_path)
    sft_loss_plot_html = _load_sft_loss_plot_html(mid)
    if has_sft_ckpt:
        if not sft_loss_plot_html:
            sft_loss_plot_html = generate_loss_chart_html([], [])
        loss_plot_html_content = _combine_loss_plots(base_loss_plot_html, sft_loss_plot_html)
    else:
        loss_plot_html_content = base_loss_plot_html

    inference_history = dbm.get_inference_history(mid) or ""

    display_params = {}
    try:
        if cfg:
            display_params = {
                "Model Structure": {
                    "layers": cfg.get("n_layer"),
                    "heads": cfg.get("n_head"),
                    "embedding_dim": cfg.get("n_embd"),
                    "block_size": cfg.get("block_size"),
                    "dropout": cfg.get("dropout"),
                    "bias": cfg.get("bias"),
                },
                "Training": {
                    "learning_rate": cfg.get("learning_rate"),
                    "batch_size": cfg.get("batch_size"),
                    "iterations": cfg.get("max_iters"),
                    "scheduler": cfg.get("lr_scheduler_type"),
                },
            }

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
                    "rope_base": cfg.get("rope_base"),
                }

        if sft_cfg:
            display_params["SFT"] = {
                "epochs": sft_cfg.get("epochs"),
                "learning_rate": sft_cfg.get("learning_rate"),
                "batch_size": sft_cfg.get("batch_size"),
                "max_seq_length": sft_cfg.get("max_seq_length"),
                "gradient_accumulation_steps": sft_cfg.get("gradient_accumulation_steps"),
                "lr_scheduler_type": sft_cfg.get("lr_scheduler_type"),
                "warmup_iters": sft_cfg.get("warmup_iters"),
                "lr_decay_iters": sft_cfg.get("lr_decay_iters"),
                "min_lr": sft_cfg.get("min_lr"),
                "step_size": sft_cfg.get("step_size"),
                "step_gamma": sft_cfg.get("step_gamma"),
                "polynomial_power": sft_cfg.get("polynomial_power"),
                "label_smoothing": sft_cfg.get("label_smoothing"),
                "freeze_layers": sft_cfg.get("freeze_layers"),
                "grad_clip": sft_cfg.get("grad_clip"),
                "weight_decay": sft_cfg.get("weight_decay"),
                "system_prompt": sft_cfg.get("system_prompt"),
            }
    except Exception as e:
        print(f"Error formatting parameters: {e}")

    return [
        display_params,
        loss_plot_html_content,
        inference_history,
        data_processed_dir,
        out_dir_root,
        has_sft_ckpt,
    ]


def dual_inference_cb(
    left_data_dir, left_out_dir,
    right_data_dir, right_out_dir,
    left_model_sel, right_model_sel,
    prompt,
    left_num_samples, left_max_tokens, left_temperature, left_top_k, left_dtype, left_seed,
    right_num_samples, right_max_tokens, right_temperature, right_top_k, right_dtype, right_seed,
):
    """
    Optimized dual model concurrent inference using caching system and improved concurrency strategy
    """
    print("🔥 Starting dual model comparison inference...")

    def _resolve_model_paths(model_sel, data_dir, out_dir):
        if data_dir and out_dir:
            ckpt_path = out_dir if out_dir.endswith(".pt") else os.path.join(out_dir, "ckpt.pt")
            if os.path.exists(ckpt_path):
                return data_dir, out_dir
        if not model_sel or " - " not in model_sel:
            return data_dir, out_dir
        try:
            model_id = int(model_sel.split(" - ")[0])
            model_info = dbm.get_model(model_id)
            if model_info:
                data_dir = data_dir or model_info.get("processed_data_dir", data_dir)
                out_dir = out_dir or model_info.get("out_dir", out_dir)
        except Exception:
            pass
        return data_dir, out_dir

    left_data_dir, left_out_dir = _resolve_model_paths(left_model_sel, left_data_dir, left_out_dir)
    right_data_dir, right_out_dir = _resolve_model_paths(right_model_sel, right_data_dir, right_out_dir)

    if not left_out_dir or not right_out_dir:
        error_html = _render_error_html("Please select two models for comparison first.")
        yield error_html, error_html, gr.update(interactive=True), gr.update(interactive=False)
        return

    if not prompt.strip():
        error_html = _render_error_html("Prompt is empty, please enter starting text.")
        yield error_html, error_html, gr.update(interactive=True), gr.update(interactive=False)
        return

    left_html = ""
    right_html = ""
    cache = None
    yield gr.update(), gr.update(), gr.update(interactive=False), gr.update(interactive=True)

    left_prompt_tokens = _tokenize_prompt_for_display(left_data_dir, prompt)
    right_prompt_tokens = _tokenize_prompt_for_display(right_data_dir, prompt)
    left_state = _init_stream_state(prompt, left_prompt_tokens)
    right_state = _init_stream_state(prompt, right_prompt_tokens)

    try:
        try:
            left_params = {
                "num_samples": int(float(left_num_samples)),
                "max_tokens": int(float(left_max_tokens)),
                "temperature": float(left_temperature),
                "top_k": int(float(left_top_k)) if left_top_k is not None and str(left_top_k).strip() != "" else None,
                "seed": int(float(left_seed)),
                "dtype": left_dtype,
            }
        except ValueError as e:
            error_html = _render_error_html(f"Left model parameter error: {str(e)}")
            yield error_html, right_html, gr.update(interactive=True), gr.update(interactive=False)
            return

        try:
            right_params = {
                "num_samples": int(float(right_num_samples)),
                "max_tokens": int(float(right_max_tokens)),
                "temperature": float(right_temperature),
                "top_k": int(float(right_top_k)) if right_top_k is not None and str(right_top_k).strip() != "" else None,
                "seed": int(float(right_seed)),
                "dtype": right_dtype,
            }
        except ValueError as e:
            error_html = _render_error_html(f"Right model parameter error: {str(e)}")
            yield left_html, error_html, gr.update(interactive=True), gr.update(interactive=False)
            return

        cache = ModelCache()
        cache_info = cache.get_cache_info()
        print(f"🔍 Comparison inference started - Cache status: {cache_info}")

        left_ckpt_path = left_out_dir if left_out_dir.endswith(".pt") else os.path.join(left_out_dir, "ckpt.pt")
        right_ckpt_path = right_out_dir if right_out_dir.endswith(".pt") else os.path.join(right_out_dir, "ckpt.pt")

        if not os.path.exists(left_ckpt_path):
            error_html = _render_error_html(f"Left model checkpoint not found: {left_ckpt_path}")
            yield error_html, right_html, gr.update(interactive=True), gr.update(interactive=False)
            return

        if not os.path.exists(right_ckpt_path):
            error_html = _render_error_html(f"Right model checkpoint not found: {right_ckpt_path}")
            yield left_html, error_html, gr.update(interactive=True), gr.update(interactive=False)
            return

        print("✅ Both checkpoint files verified")

        try:
            left_checkpoint = torch.load(left_ckpt_path, map_location="cpu")
            left_model_args = left_checkpoint["model_args"]
            left_model_type = cache._detect_model_type(left_model_args)

            right_checkpoint = torch.load(right_ckpt_path, map_location="cpu")
            right_model_args = right_checkpoint["model_args"]
            right_model_type = cache._detect_model_type(right_model_args)

            print(f"🔍 Model types detected - Left: {left_model_type}, Right: {right_model_type}")

            left_vocab_size = left_model_args.get("vocab_size", 0)
            right_vocab_size = right_model_args.get("vocab_size", 0)

            if left_vocab_size != right_vocab_size:
                print(f"⚠️ Warning: Different vocab sizes - Left: {left_vocab_size}, Right: {right_vocab_size}")

            try:
                left_model, left_gptconf, left_encode, left_decode = cache.get_model_and_meta(
                    left_ckpt_path, left_data_dir, "cpu", left_dtype
                )
                print(f"✅ Left model ({left_model_type}) loaded successfully")

                right_model, right_gptconf, right_encode, right_decode = cache.get_model_and_meta(
                    right_ckpt_path, right_data_dir, "cpu", right_dtype
                )
                print(f"✅ Right model ({right_model_type}) loaded successfully")

                del left_model, right_model
                import gc

                gc.collect()

            except Exception as model_load_error:
                error_html = _render_error_html(f"Model loading test failed: {str(model_load_error)}")
                print(f"❌ {error_html}")
                yield error_html, error_html, gr.update(interactive=True), gr.update(interactive=False)
                return

        except Exception as e:
            print(f"⚠️ Warning: Model compatibility check failed: {e}")

        left_memory_req = 0
        right_memory_req = 0

        if os.path.exists(left_ckpt_path):
            left_size_mb = os.path.getsize(left_ckpt_path) / (1024 * 1024)
            left_memory_req = device_manager.estimate_model_memory(left_size_mb)

        if os.path.exists(right_ckpt_path):
            right_size_mb = os.path.getsize(right_ckpt_path) / (1024 * 1024)
            right_memory_req = device_manager.estimate_model_memory(right_size_mb)

        left_device, right_device = device_manager.allocate_devices_for_comparison(
            left_memory_req, right_memory_req
        )

        print(f"🎯 Device allocation: left model={left_device}, right model={right_device}")

        left_queue = queue.Queue(maxsize=1000)
        right_queue = queue.Queue(maxsize=1000)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ModelInference") as executor:

            def run_cached_inference(data_dir, out_dir, params, result_queue, model_name, assigned_device):
                try:
                    gen = cached_generate_text(
                        data_dir=data_dir,
                        out_dir=out_dir,
                        prompt=prompt,
                        num_samples=params["num_samples"],
                        max_new_tokens=params["max_tokens"],
                        temperature=params["temperature"],
                        top_k=params["top_k"],
                        seed=params["seed"],
                        device=assigned_device,
                        dtype=params["dtype"],
                        compile_model=DEFAULT_CONFIG["inference"]["compile_model"],
                        auto_clear_cache=False,
                        return_detailed_info=True,
                    )

                    for piece in gen:
                        text_piece, _ = piece
                        # Check if this is an error message from cached_generate_text
                        if text_piece.startswith("Error:"):
                            result_queue.put(("error", text_piece))
                            result_queue.put(("done", None))
                            return
                        result_queue.put(("data", piece))

                    result_queue.put(("done", None))

                except UnknownTokenError as e:
                    error_msg = (
                        f"❌ {model_name} Error: {str(e)}\n\nInference aborted. Please ensure your prompt only contains characters/tokens present in the training vocabulary."
                    )
                    print(f"❌ {model_name} Unknown token error: {str(e)}")
                    result_queue.put(("error", error_msg))
                    result_queue.put(("done", None))

                except Exception as e:
                    error_msg = f"{model_name} generation error: {str(e)}"
                    print(f"❌ {error_msg}")
                    import traceback

                    print(traceback.format_exc())
                    result_queue.put(("error", error_msg))
                    result_queue.put(("done", None))

            left_future = executor.submit(
                run_cached_inference,
                left_data_dir,
                left_out_dir,
                left_params,
                left_queue,
                "Left model",
                left_device,
            )

            right_future = executor.submit(
                run_cached_inference,
                right_data_dir,
                right_out_dir,
                right_params,
                right_queue,
                "Right model",
                right_device,
            )

            left_done = False
            right_done = False
            last_yield_time = 0
            min_yield_interval = 0.05

            while not (left_done and right_done):
                current_time = time.time()
                updated = False

                left_updated = False
                while not left_done:
                    try:
                        msg_type, data = left_queue.get_nowait()
                        if msg_type == "data":
                            text_piece, token_detail = data
                            _update_stream_state(left_state, text_piece, token_detail)
                            left_updated = True
                        elif msg_type == "error":
                            left_html = _render_error_html(data)
                            updated = True
                        elif msg_type == "done":
                            left_done = True
                            break
                    except queue.Empty:
                        break

                if left_updated:
                    left_html = _render_stream_html(left_state)
                    updated = True

                right_updated = False
                while not right_done:
                    try:
                        msg_type, data = right_queue.get_nowait()
                        if msg_type == "data":
                            text_piece, token_detail = data
                            _update_stream_state(right_state, text_piece, token_detail)
                            right_updated = True
                        elif msg_type == "error":
                            right_html = _render_error_html(data)
                            updated = True
                        elif msg_type == "done":
                            right_done = True
                            break
                    except queue.Empty:
                        break

                if right_updated:
                    right_html = _render_stream_html(right_state)
                    updated = True

                if updated and (current_time - last_yield_time) >= min_yield_interval:
                    yield left_html, right_html, gr.update(interactive=False), gr.update(interactive=True)
                    last_yield_time = current_time
                elif not updated:
                    time.sleep(0.02)

            try:
                left_future.result(timeout=10.0)
                right_future.result(timeout=10.0)
                print("✅ Both inference tasks completed")
            except concurrent.futures.TimeoutError:
                print("⚠️ Warning: Some inference tasks may not have completed properly")
            except Exception as e:
                print(f"⚠️ Warning: Task completion error: {e}")

            _finalize_stream_state(left_state)
            _finalize_stream_state(right_state)
            left_html = _render_stream_html(left_state)
            right_html = _render_stream_html(right_state)
            yield left_html, right_html, gr.update(interactive=True), gr.update(interactive=False)

            print("🏁 Comparison inference completed successfully")

    except Exception as e:
        error_msg = f"Comparison inference error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback

        print(traceback.format_exc())

        if not left_html.strip():
            left_html = _render_error_html(error_msg)
        if not right_html.strip():
            right_html = _render_error_html(error_msg)

        yield left_html, right_html, gr.update(interactive=True), gr.update(interactive=False)

    finally:
        try:
            if cache is None:
                cache = ModelCache()
            cache.clear_cache()
            print("🧹 Comparison inference completed, cache cleared for optimal performance")
        except Exception as cleanup_error:
            print(f"Warning: Cache cleanup failed: {cleanup_error}")


def dual_chat_cb(
    user_msg,
    left_history,
    right_history,
    left_model_sel,
    right_model_sel,
    sys_prompt,
    left_max_tokens,
    left_temperature,
    left_top_k,
    left_seed,
    right_max_tokens,
    right_temperature,
    right_top_k,
    right_seed,
):
    import gradio as gr

    if not user_msg or not str(user_msg).strip():
        gr.Warning("⚠️ Please enter a message.")
        return "", left_history or [], right_history or []

    left_history = left_history or []
    right_history = right_history or []

    if not left_model_sel or " - " not in left_model_sel or not right_model_sel or " - " not in right_model_sel:
        left_history.append({"role": "assistant", "content": "❌ Please select two models first."})
        right_history.append({"role": "assistant", "content": "❌ Please select two models first."})
        return "", left_history, right_history

    left_model_id = int(left_model_sel.split(" - ")[0])
    right_model_id = int(right_model_sel.split(" - ")[0])

    left_info = dbm.get_model(left_model_id)
    right_info = dbm.get_model(right_model_id)
    if not left_info or not right_info:
        left_history.append({"role": "assistant", "content": "❌ Model not found."})
        right_history.append({"role": "assistant", "content": "❌ Model not found."})
        return "", left_history, right_history

    left_ckpt_path = os.path.join(left_info["out_dir"], "sft", "ckpt_sft.pt")
    right_ckpt_path = os.path.join(right_info["out_dir"], "sft", "ckpt_sft.pt")

    if not os.path.exists(left_ckpt_path) or not os.path.exists(right_ckpt_path):
        left_history.append({"role": "assistant", "content": "❌ Both models must have SFT checkpoints for chat mode."})
        right_history.append({"role": "assistant", "content": "❌ Both models must have SFT checkpoints for chat mode."})
        return "", left_history, right_history

    tokenizer_path = Path.cwd() / "assets" / "tokenizer.json"
    if not tokenizer_path.exists():
        left_history.append({"role": "assistant", "content": "❌ tokenizer.json not found."})
        right_history.append({"role": "assistant", "content": "❌ tokenizer.json not found."})
        return "", left_history, right_history

    from tokenizers import Tokenizer

    left_tokenizer = Tokenizer.from_file(str(tokenizer_path))
    right_tokenizer = Tokenizer.from_file(str(tokenizer_path))

    left_old2new = _load_old2new_mapping(left_info.get("processed_data_dir", ""))
    right_old2new = _load_old2new_mapping(right_info.get("processed_data_dir", ""))

    if left_old2new is None or right_old2new is None:
        left_history.append({"role": "assistant", "content": "❌ Token ID mapping not found. Please ensure models were trained with the custom tokenizer."})
        right_history.append({"role": "assistant", "content": "❌ Token ID mapping not found. Please ensure models were trained with the custom tokenizer."})
        return "", left_history, right_history

    left_new2old = {new_id: old_id for old_id, new_id in left_old2new.items()}
    right_new2old = {new_id: old_id for old_id, new_id in right_old2new.items()}

    left_user_tokens_info = tokenize_user_input(left_tokenizer, user_msg, left_old2new)
    right_user_tokens_info = tokenize_user_input(right_tokenizer, user_msg, right_old2new)

    left_user_msg_html = (
        f'<div style="font-family: system-ui, sans-serif;">{_escape_html(user_msg)}'
        f"{_generate_user_tokenization_html(left_user_tokens_info)}</div>"
    )
    right_user_msg_html = (
        f'<div style="font-family: system-ui, sans-serif;">{_escape_html(user_msg)}'
        f"{_generate_user_tokenization_html(right_user_tokens_info)}</div>"
    )

    left_history.append({"role": "user", "content": left_user_msg_html})
    right_history.append({"role": "user", "content": right_user_msg_html})

    yield "", left_history, right_history

    left_messages = _build_messages_from_history(left_history)
    right_messages = _build_messages_from_history(right_history)

    left_params = {
        "max_tokens": int(float(left_max_tokens)),
        "temperature": float(left_temperature),
        "top_k": int(float(left_top_k)) if left_top_k is not None and str(left_top_k).strip() != "" else 0,
        "seed": int(float(left_seed)),
    }
    right_params = {
        "max_tokens": int(float(right_max_tokens)),
        "temperature": float(right_temperature),
        "top_k": int(float(right_top_k)) if right_top_k is not None and str(right_top_k).strip() != "" else 0,
        "seed": int(float(right_seed)),
    }

    left_memory_req = 0
    right_memory_req = 0
    if os.path.exists(left_ckpt_path):
        left_size_mb = os.path.getsize(left_ckpt_path) / (1024 * 1024)
        left_memory_req = device_manager.estimate_model_memory(left_size_mb)
    if os.path.exists(right_ckpt_path):
        right_size_mb = os.path.getsize(right_ckpt_path) / (1024 * 1024)
        right_memory_req = device_manager.estimate_model_memory(right_size_mb)

    left_device, right_device = device_manager.allocate_devices_for_comparison(
        left_memory_req, right_memory_req
    )

    left_queue = queue.Queue(maxsize=1000)
    right_queue = queue.Queue(maxsize=1000)

    left_state = {"history": left_history, "response_tokens": [], "assistant_started": False}
    right_state = {"history": right_history, "response_tokens": [], "assistant_started": False}

    def _update_chat_history(state, text_piece, token_detail):
        state["response_tokens"].append({"text": text_piece, "token_detail": token_detail})
        response_html = (
            f'<div style="font-family: system-ui, sans-serif;">'
            f"{_generate_response_html_with_tokens(state['response_tokens'])}</div>"
        )
        if state["assistant_started"]:
            state["history"][-1] = {"role": "assistant", "content": response_html}
        else:
            state["history"].append({"role": "assistant", "content": response_html})
            state["assistant_started"] = True

    def _apply_chat_error(state, message):
        error_html = f"<div style='color: red;'>{_escape_html(message)}</div>"
        if state["assistant_started"]:
            state["history"][-1] = {"role": "assistant", "content": error_html}
        else:
            state["history"].append({"role": "assistant", "content": error_html})
            state["assistant_started"] = True

    def run_chat_generation(
        ckpt_path,
        tokenizer,
        messages,
        system_prompt,
        params,
        old2new,
        new2old,
        assigned_device,
        result_queue,
        model_name,
    ):
        model = None
        try:
            checkpoint = torch.load(ckpt_path, map_location=assigned_device)
            model_args = checkpoint["model_args"]
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
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            model.load_state_dict(state_dict)
            model.to(assigned_device)
            model.eval()

            generator = chat_generate(
                model=model,
                tokenizer=tokenizer,
                messages=messages,
                system_prompt=system_prompt,
                max_new_tokens=params["max_tokens"],
                temperature=params["temperature"],
                top_k=params["top_k"],
                old2new_mapping=old2new,
                new2old_mapping=new2old,
                return_detailed_info=True,
                history_token_ids=None,
                seed=params["seed"],
            )

            for item in generator:
                text_piece, token_detail = item
                if token_detail and token_detail.get("is_final"):
                    continue
                if not text_piece:
                    continue
                result_queue.put(("data", text_piece, token_detail))

            result_queue.put(("done", None, None))

        except Exception as e:
            error_msg = f"{model_name} chat error: {str(e)}"
            print(error_msg)
            result_queue.put(("error", error_msg, None))
            result_queue.put(("done", None, None))
        finally:
            try:
                if model is not None:
                    del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as cleanup_error:
                print(f"Warning: Chat cleanup failed: {cleanup_error}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="ModelChat") as executor:
        left_future = executor.submit(
            run_chat_generation,
            left_ckpt_path,
            left_tokenizer,
            left_messages,
            sys_prompt,
            left_params,
            left_old2new,
            left_new2old,
            left_device,
            left_queue,
            "Left model",
        )

        right_future = executor.submit(
            run_chat_generation,
            right_ckpt_path,
            right_tokenizer,
            right_messages,
            sys_prompt,
            right_params,
            right_old2new,
            right_new2old,
            right_device,
            right_queue,
            "Right model",
        )

        left_done = False
        right_done = False
        last_yield_time = 0
        min_yield_interval = 0.05

        while not (left_done and right_done):
            current_time = time.time()
            updated = False

            while not left_done:
                try:
                    msg_type, data, detail = left_queue.get_nowait()
                    if msg_type == "data":
                        _update_chat_history(left_state, data, detail)
                        updated = True
                    elif msg_type == "error":
                        _apply_chat_error(left_state, data)
                        updated = True
                    elif msg_type == "done":
                        left_done = True
                        break
                except queue.Empty:
                    break

            while not right_done:
                try:
                    msg_type, data, detail = right_queue.get_nowait()
                    if msg_type == "data":
                        _update_chat_history(right_state, data, detail)
                        updated = True
                    elif msg_type == "error":
                        _apply_chat_error(right_state, data)
                        updated = True
                    elif msg_type == "done":
                        right_done = True
                        break
                except queue.Empty:
                    break

            if updated and (current_time - last_yield_time) >= min_yield_interval:
                yield "", left_state["history"], right_state["history"]
                last_yield_time = current_time
            elif not updated:
                time.sleep(0.02)

        try:
            left_future.result(timeout=10.0)
            right_future.result(timeout=10.0)
        except Exception as e:
            print(f"⚠️ Warning: Chat task completion error: {e}")

        yield "", left_state["history"], right_state["history"]


def clear_compare_chat_cb():
    stop_chat_generation()
    return "", [], []


def stop_comparison_cb():
    stop_inference()
    return gr.update(), gr.update(), gr.update(interactive=True), gr.update(interactive=False)
