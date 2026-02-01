import concurrent.futures
import os
import queue
import time

import torch

from src.config import DEFAULT_CONFIG
from src.device_manager import device_manager
from src.infer_cache import cached_generate_text, ModelCache, UnknownTokenError
from src.ui.charts import generate_loss_chart_html
from src.ui.helpers import _create_plot_html_from_log
from src.ui.state import dbm


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
    info = dbm.get_model_basic_info(mid) or {}
    name = info.get("name", "unknown_model")

    # Use relative paths for portability
    if "dir_path" in info:
        out_dir_root = info["dir_path"]
        folder = os.path.basename(out_dir_root)
        data_processed_dir = os.path.join("data", folder, "processed")
    else:
        folder_name_part = "".join(c if c.isalnum() or c in ["_", "-"] else "_" for c in name)
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
                    "bias": cfg.get("bias"),
                },
                "Training": {
                    "learning_rate": cfg.get("learning_rate"),
                    "batch_size": cfg.get("batch_size"),
                    "iterations": cfg.get("max_iters"),
                    "scheduler": cfg.get("lr_scheduler_type"),
                },
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
                    "rope_base": cfg.get("rope_base"),
                }
        except Exception as e:
            print(f"Error formatting parameters: {e}")

    return [display_params, loss_plot_html_content, inference_history, data_processed_dir, out_dir_root]


def dual_inference_cb(
    left_data_dir, left_out_dir,
    right_data_dir, right_out_dir,
    prompt,
    left_num_samples, left_max_tokens, left_temperature, left_top_k, left_dtype, left_seed,
    right_num_samples, right_max_tokens, right_temperature, right_top_k, right_dtype, right_seed,
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
                "num_samples": int(float(left_num_samples)),
                "max_tokens": int(float(left_max_tokens)),
                "temperature": float(left_temperature),
                "top_k": int(float(left_top_k)) if left_top_k is not None and str(left_top_k).strip() != "" else None,
                "seed": int(float(left_seed)),
                "dtype": left_dtype,
            }
        except ValueError as e:
            error_msg = f"Left model parameter error: {str(e)}"
            yield error_msg, right_output
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
            error_msg = f"Right model parameter error: {str(e)}"
            yield left_output, error_msg
            return

        # Get cache instance to optimize model loading
        cache = ModelCache()
        cache_info = cache.get_cache_info()
        print(f"🔍 Comparison inference started - Cache status: {cache_info}")

        # Verify checkpoint files exist and get model info
        left_ckpt_path = left_out_dir if left_out_dir.endswith(".pt") else os.path.join(left_out_dir, "ckpt.pt")
        right_ckpt_path = right_out_dir if right_out_dir.endswith(".pt") else os.path.join(right_out_dir, "ckpt.pt")

        if not os.path.exists(left_ckpt_path):
            error_msg = f"Left model checkpoint not found: {left_ckpt_path}"
            yield error_msg, right_output
            return

        if not os.path.exists(right_ckpt_path):
            error_msg = f"Right model checkpoint not found: {right_ckpt_path}"
            yield left_output, error_msg
            return

        print("✅ Both checkpoint files verified")

        # Pre-check model types and compatibility
        try:
            left_checkpoint = torch.load(left_ckpt_path, map_location="cpu")
            left_model_args = left_checkpoint["model_args"]
            left_model_type = cache._detect_model_type(left_model_args)

            right_checkpoint = torch.load(right_ckpt_path, map_location="cpu")
            right_model_args = right_checkpoint["model_args"]
            right_model_type = cache._detect_model_type(right_model_args)

            print(f"🔍 Model types detected - Left: {left_model_type}, Right: {right_model_type}")

            # Check vocab size compatibility for comparison
            left_vocab_size = left_model_args.get("vocab_size", 0)
            right_vocab_size = right_model_args.get("vocab_size", 0)

            if left_vocab_size != right_vocab_size:
                print(f"⚠️ Warning: Different vocab sizes - Left: {left_vocab_size}, Right: {right_vocab_size}")

            # Test model loading capability
            try:
                left_model, left_gptconf, left_encode, left_decode = cache.get_model_and_meta(
                    left_ckpt_path, left_data_dir, "cpu", left_dtype
                )
                print(f"✅ Left model ({left_model_type}) loaded successfully")

                right_model, right_gptconf, right_encode, right_decode = cache.get_model_and_meta(
                    right_ckpt_path, right_data_dir, "cpu", right_dtype
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
            # Continue anyway

        # Smart device allocation
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
                """Wrapper function for running cached inference with smart device allocation"""
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
                    )

                    # Batch process generated text fragments
                    text_buffer = ""
                    buffer_size = 5

                    for piece in gen:
                        text_buffer += piece
                        if len(text_buffer) >= buffer_size:
                            result_queue.put(("data", text_buffer))
                            text_buffer = ""

                    if text_buffer:
                        result_queue.put(("data", text_buffer))

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

                # Batch process left model output
                left_batch = []
                while not left_done:
                    try:
                        msg_type, data = left_queue.get_nowait()
                        if msg_type == "data":
                            left_batch.append(data)
                        elif msg_type == "error":
                            left_output = data
                            updated = True
                            break
                        elif msg_type == "done":
                            left_done = True
                            break
                    except queue.Empty:
                        break

                if left_batch:
                    left_output += "".join(left_batch)
                    updated = True

                # Batch process right model output
                right_batch = []
                while not right_done:
                    try:
                        msg_type, data = right_queue.get_nowait()
                        if msg_type == "data":
                            right_batch.append(data)
                        elif msg_type == "error":
                            right_output = data
                            updated = True
                            break
                        elif msg_type == "done":
                            right_done = True
                            break
                    except queue.Empty:
                        break

                if right_batch:
                    right_output += "".join(right_batch)
                    updated = True

                # Control output frequency
                if updated and (current_time - last_yield_time) >= min_yield_interval:
                    yield left_output, right_output
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

            # Final output
            yield left_output, right_output

            print("🏁 Comparison inference completed successfully")

    except Exception as e:
        error_msg = f"Comparison inference error: {str(e)}"
        print(f"❌ {error_msg}")
        import traceback

        print(traceback.format_exc())

        if not left_output.strip():
            left_output = error_msg
        if not right_output.strip():
            right_output = error_msg

        yield left_output, right_output

    finally:
        # Unified cache cleanup
        try:
            if cache is None:
                cache = ModelCache()
            cache.clear_cache()
            print("🧹 Comparison inference completed, cache cleared for optimal performance")
        except Exception as cleanup_error:
            print(f"Warning: Cache cleanup failed: {cleanup_error}")
