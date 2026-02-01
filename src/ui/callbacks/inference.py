import os
from pathlib import Path

import torch

from src.config import DEFAULT_CONFIG
from src.infer_cache import cached_generate_text, ModelCache, UnknownTokenError
from src.sft import tokenize_user_input
from src.ui.html_render import _escape_html, _generate_token_html, _generate_advanced_html
from src.ui.state import dbm


def inference_cb(
    data_dir_inf_, out_dir_inf_,
    prompt_, num_samples_, max_new_tokens_,
    temperature_, top_k_, dtype_inf_, device_inf_, seed_inf_,
):
    cache = None
    prompt_tokens = None
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
            meta_path = os.path.join(data_dir_inf_, "meta.pkl")
            if os.path.exists(meta_path):
                import pickle

                with open(meta_path, "rb") as f:
                    meta = pickle.load(f)
                old2new_mapping = meta.get("old2new_mapping") or meta.get("old2new")
                tokenizer_type = meta.get("tokenizer_type") or meta.get("tokenizer", "char_level")

                if tokenizer_type == "custom_json" and old2new_mapping:
                    tokenizer_path = Path.cwd() / "assets" / "tokenizer.json"
                    if tokenizer_path.exists():
                        from tokenizers import Tokenizer

                        tokenizer = Tokenizer.from_file(str(tokenizer_path))
                        prompt_tokens = tokenize_user_input(tokenizer, prompt_, old2new_mapping)
                elif tokenizer_type == "gpt2" and old2new_mapping:
                    import tiktoken

                    enc = tiktoken.get_encoding("gpt2")
                    ids = enc.encode(prompt_, allowed_special={"<|endoftext|>"})
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
                    # Character level tokenization
                    stoi = meta.get("stoi", {})
                    prompt_tokens = []
                    for ch in prompt_:
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

        # Use cached inference function with detailed info enabled
        gen = cached_generate_text(
            data_dir=data_dir_inf_,
            out_dir=out_dir_inf_,
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
            return_detailed_info=True,
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
                    all_samples_info.append(
                        {
                            "sample_idx": current_sample_idx - 1,
                            "prompt": prompt_,
                            "tokens": current_sample_tokens.copy(),
                        }
                    )
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
                    all_samples_info.append(
                        {
                            "sample_idx": current_sample_idx - 1,
                            "prompt": prompt_,
                            "tokens": current_sample_tokens.copy(),
                        }
                    )
                current_sample_tokens = []
                full_text_output += text_piece
            else:
                # Generated token
                current_sample_tokens.append({"text": text_piece, "token_detail": token_detail})
                if token_detail:
                    all_token_details.append(
                        {
                            "sample_index": current_sample_idx - 1,
                            "token_details": [token_detail],
                        }
                    )
                full_text_output += text_piece

            # Generate HTML outputs
            # Main output with token highlighting
            main_html_parts = []
            main_html_parts.append('<div style="font-family: system-ui, sans-serif; background: #ffffff; border: 2px solid #000000; border-radius: 8px; padding: 15px;">')

            # Show all completed samples
            for sample_info in all_samples_info:
                main_html_parts.append(
                    f'<div style="margin-bottom: 15px;"><strong>Sample {sample_info["sample_idx"] + 1}:</strong><br>'
                )
                main_html_parts.append(_generate_token_html(sample_info["tokens"], prompt_tokens=prompt_tokens))
                main_html_parts.append("</div>")

            # Show current sample in progress
            if current_sample_tokens or prompt_displayed:
                main_html_parts.append(
                    f'<div style="margin-bottom: 15px;"><strong>Sample {current_sample_idx}:</strong><br>'
                )
                main_html_parts.append(
                    _generate_token_html(current_sample_tokens, prompt_tokens=prompt_tokens if prompt_displayed else None)
                )
                main_html_parts.append("</div>")

            main_html_parts.append("</div>")
            main_html = "".join(main_html_parts)

            # Advanced output - consolidate token details by sample
            consolidated_details = {}
            for td in all_token_details:
                s_idx = td["sample_index"]
                if s_idx not in consolidated_details:
                    consolidated_details[s_idx] = {"sample_index": s_idx, "token_details": []}
                consolidated_details[s_idx]["token_details"].extend(td["token_details"])

            advanced_html = _generate_advanced_html(list(consolidated_details.values()))

            yield main_html, advanced_html

        # Final update - add the last sample
        if current_sample_tokens:
            all_samples_info.append(
                {
                    "sample_idx": current_sample_idx - 1,
                    "prompt": prompt_,
                    "tokens": current_sample_tokens.copy(),
                }
            )

        # Final HTML output
        main_html_parts = []
        main_html_parts.append('<div style="font-family: system-ui, sans-serif; background: #ffffff; border: 2px solid #000000; border-radius: 8px; padding: 15px;">')
        for sample_info in all_samples_info:
            main_html_parts.append(
                f'<div style="margin-bottom: 15px;"><strong>Sample {sample_info["sample_idx"] + 1}:</strong><br>'
            )
            main_html_parts.append(_generate_token_html(sample_info["tokens"], prompt_tokens=prompt_tokens))
            main_html_parts.append("</div>")
        main_html_parts.append("</div>")
        main_html = "".join(main_html_parts)

        consolidated_details = {}
        for td in all_token_details:
            s_idx = td["sample_index"]
            if s_idx not in consolidated_details:
                consolidated_details[s_idx] = {"sample_index": s_idx, "token_details": []}
            consolidated_details[s_idx]["token_details"].extend(td["token_details"])

        advanced_html = _generate_advanced_html(list(consolidated_details.values()))

        # Save HTML formatted output to database for persistence across page reloads
        try:
            ckpt_dir = out_dir_inf_ if out_dir_inf_.endswith(".pt") else os.path.join(out_dir_inf_, "ckpt.pt")
            model_dir_for_db = os.path.dirname(ckpt_dir)
            model_id = dbm.get_model_id_by_dir(model_dir_for_db)
            if model_id:
                # Generate plain text content for backward compatibility
                plain_text_parts = []
                for sample_info in all_samples_info:
                    sample_text = f"Sample {sample_info['sample_idx'] + 1}:\n"
                    sample_text += prompt_
                    sample_text += "".join([t.get("text", "") for t in sample_info["tokens"]])
                    plain_text_parts.append(sample_text)
                plain_text = "\n\n".join(plain_text_parts)

                dbm.save_inference_history(model_id, plain_text, main_html, advanced_html)
                print(f"💾 Inference history saved to database (model_id={model_id})")
        except Exception as save_err:
            print(f"Warning: Failed to save inference history to database: {save_err}")

        yield main_html, advanced_html

        print("✅ Single model inference completed successfully")

    except UnknownTokenError as e:
        error_msg = (
            f"❌ Error: {str(e)}<br><br>Inference aborted. Please ensure your prompt only contains characters/tokens present in the training vocabulary."
        )
        print(f"❌ Unknown token error: {e}")
        error_html = f"<div style='color: red; padding: 10px; background: #ffe6e6; border-radius: 8px;'>{error_msg}</div>"
        yield error_html, ""

    except Exception as e:
        import traceback

        error_msg = f"Error during inference: {str(e)}"
        print(f"❌ Single inference error: {error_msg}")
        print(traceback.format_exc())
        error_html = f"<div style='color: red; padding: 10px; background: #ffe6e6; border-radius: 8px;'>{_escape_html(error_msg)}</div>"
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
