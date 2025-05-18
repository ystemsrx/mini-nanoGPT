# src/ui.py
import os
import pickle
import io

import numpy as np
import torch
import torch._dynamo
torch._dynamo.config.suppress_errors = True
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.config import DEFAULT_CONFIG, LANG_JSON, IntegerTypes
from src.db_manager import DBManager
from src.data_process import process_data
from src.train import train_model_generator, stop_training
from src.infer import generate_text

dbm = DBManager()

def _model_choices():
    return [f"{m['id']} - {m['name']}" for m in dbm.get_all_models()]

def _load_loss_plot(loss_log_path: str):
    if not (loss_log_path and os.path.exists(loss_log_path)):
        return None
    try:
        with open(loss_log_path, "rb") as f:
            loss_dict = pickle.load(f)
        tr_steps = loss_dict.get("train_plot_steps", [])
        tr_losses = loss_dict.get("train_plot_losses", [])
        val_steps = loss_dict.get("val_plot_steps", [])
        val_losses = loss_dict.get("val_plot_losses", [])
        if not tr_steps:
            return None
        fig, ax = plt.subplots()
        ax.plot(tr_steps, tr_losses, label="train")
        if val_losses:
            ax.plot(val_steps, val_losses, label="val")
        ax.set_xlabel("Iter"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)
        buf = io.BytesIO(); plt.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
        return Image.open(buf)
    except Exception:
        return None

def build_app_interface(selected_lang: str = "zh"):
    """
        Top-level UI function
        Implemented:
          · Logic for new model/model name, automatic directory, dropdown refresh/delete  
          · Language switching: After switching the `lang_select` dropdown, **all component labels & default values** are refreshed synchronously  
    """

    # ------------------------------------------------------------------ #
    # tools
    # ------------------------------------------------------------------ #
    def _model_choices():
        return [f"{m['id']} - {m['name']}" for m in dbm.get_all_models()]

    def _load_loss_plot(loss_log_path: str):
        if not (loss_log_path and os.path.exists(loss_log_path)):
            return None
        try:
            with open(loss_log_path, "rb") as f:
                loss_dict = pickle.load(f)
            tr_steps = loss_dict.get("train_plot_steps", [])
            tr_losses = loss_dict.get("train_plot_losses", [])
            val_steps = loss_dict.get("val_plot_steps", [])
            val_losses = loss_dict.get("val_plot_losses", [])
            if not tr_steps:
                return None
            fig, ax = plt.subplots()
            ax.plot(tr_steps, tr_losses, label="train")
            if val_losses:
                ax.plot(val_steps, val_losses, label="val")
            ax.set_xlabel("Iter"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)
            buf = io.BytesIO(); plt.savefig(buf, format="png"); plt.close(fig); buf.seek(0)
            return Image.open(buf)
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    # Initialize Gradio app
    # ------------------------------------------------------------------ #
    T = LANG_JSON[selected_lang]

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

    # ------------------------------------------------------------------ #
    # —— Blocks
    # ------------------------------------------------------------------ #
    with gr.Blocks(title=T["app_title"], css=custom_css) as demo:

        # ========= Top: model management / language ========= #
        with gr.Row():
            model_dropdown     = gr.Dropdown(label=T["registered_models"], choices=_model_choices(), interactive=True)
            refresh_models_btn = gr.Button(T["refresh_tables"])
            delete_model_btn   = gr.Button(T["delete_selected_model"], variant="stop")

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
                        txt_dir        = gr.Textbox(label=T["dp_txt_dir"], value="")
                        new_model_chk  = gr.Checkbox(label=T["new_model"], value=True)
                        model_name_box = gr.Textbox(label=T["model_name"], value="new_model")
                        with gr.Row():
                            no_val_set = gr.Checkbox(label=T["dp_no_val_set"],
                                                     value=DEFAULT_CONFIG["data_process"]["no_validation"])
                            use_gpt2   = gr.Checkbox(label=T["dp_use_gpt2_tokenizer"],
                                                     value=DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"])
                        train_split = gr.Slider(label=T["dp_train_split"],
                                                minimum=0.1, maximum=0.99, step=0.01,
                                                value=DEFAULT_CONFIG["data_process"]["train_split_ratio"])
                        num_proc    = gr.Number(label=T["dp_num_proc"],
                                                value=DEFAULT_CONFIG["data_process"]["num_proc"],
                                                precision=0)
                process_btn    = gr.Button(T["dp_start_btn"])
                process_output = gr.Textbox(label=T["dp_result"], lines=5, interactive=False)

            # -------------- Training Tab -------------- #
            with gr.Tab(T["train_tab"]) as train_tab:
                train_params_title_md = gr.Markdown(f"### {T['train_params_title']}")

                with gr.Row():
                    data_dir_box = gr.Textbox(label=T["train_data_dir"], value="", interactive=False)
                    out_dir_box  = gr.Textbox(label=T["train_out_dir"],  value="", interactive=False)
                    backend_box  = gr.Textbox(label=T["train_backend"],  value=DEFAULT_CONFIG["training"]["backend"])
                    device_box   = gr.Dropdown(label=T["train_device"],  choices=["cpu","cuda"],
                                               value=DEFAULT_CONFIG["training"]["device"])
                    dtype_box    = gr.Dropdown(label=T["train_dtype"],   choices=["float16","bfloat16","float32"],
                                               value=DEFAULT_CONFIG["training"]["dtype"])
                    compile_box  = gr.Checkbox(label=T["train_compile_model"],
                                               value=DEFAULT_CONFIG["training"]["compile_model"])

                with gr.Row():
                    plot_interval_box       = gr.Number(label=T["train_eval_interval"],
                                                        value=DEFAULT_CONFIG["training"]["plot_interval"])
                    log_interval_box        = gr.Number(label=T["train_log_interval"],
                                                        value=DEFAULT_CONFIG["training"]["log_interval"])
                    num_eval_seeds_box      = gr.Number(label=T["train_num_eval_seeds"],
                                                        value=DEFAULT_CONFIG["training"]["num_eval_seeds"])
                    save_best_val_ckpt_box  = gr.Checkbox(label=T["train_save_best_val_ckpt"],
                                                          value=DEFAULT_CONFIG["training"]["save_best_val_checkpoint"])
                    init_from_box           = gr.Dropdown(label=T["train_init_from"],
                                                          choices=["scratch","resume"],
                                                          value=DEFAULT_CONFIG["training"]["init_from"])
                    seed_box                = gr.Number(label=T["train_seed"],
                                                        value=DEFAULT_CONFIG["training"]["seed"])

                with gr.Row():
                    grad_acc_box  = gr.Number(label=T["train_gas"],
                                              value=DEFAULT_CONFIG["training"]["gradient_accumulation_steps"])
                    batch_size_box = gr.Number(label=T["train_batch_size"],
                                               value=DEFAULT_CONFIG["training"]["batch_size"])
                    block_size_box = gr.Number(label=T["train_block_size"],
                                               value=DEFAULT_CONFIG["training"]["block_size"])
                    n_layer_box    = gr.Number(label=T["train_n_layer"],
                                               value=DEFAULT_CONFIG["training"]["n_layer"])
                    n_head_box     = gr.Number(label=T["train_n_head"],
                                               value=DEFAULT_CONFIG["training"]["n_head"])
                    n_embd_box     = gr.Number(label=T["train_n_embd"],
                                               value=DEFAULT_CONFIG["training"]["n_embd"])

                with gr.Row():
                    dropout_box      = gr.Number(label=T["train_dropout"],
                                                 value=DEFAULT_CONFIG["training"]["dropout"])
                    bias_box         = gr.Checkbox(label=T["train_bias"],
                                                   value=DEFAULT_CONFIG["training"]["bias"])
                    lr_box           = gr.Number(label=T["train_lr"],
                                                 value=DEFAULT_CONFIG["training"]["learning_rate"])
                    max_iters_box    = gr.Number(label=T["train_max_iters"],
                                                 value=DEFAULT_CONFIG["training"]["max_iters"])
                    weight_decay_box = gr.Number(label=T["train_weight_decay"],
                                                 value=DEFAULT_CONFIG["training"]["weight_decay"])

                with gr.Row():
                    beta1_box        = gr.Number(label=T["train_beta1"],
                                                 value=DEFAULT_CONFIG["training"]["beta1"])
                    beta2_box        = gr.Number(label=T["train_beta2"],
                                                 value=DEFAULT_CONFIG["training"]["beta2"])
                    lr_scheduler_box = gr.Dropdown(label=T["train_lr_scheduler"],
                                                   choices=["none","cosine","constant_with_warmup",
                                                            "linear","step","polynomial"],
                                                   value=DEFAULT_CONFIG["training"]["lr_scheduler_type"])
                    warmup_box       = gr.Number(label=T["train_warmup_iters"],
                                                 value=DEFAULT_CONFIG["training"]["warmup_iters"])
                    lr_decay_box     = gr.Number(label=T["train_lr_decay_iters"],
                                                 value=DEFAULT_CONFIG["training"]["lr_decay_iters"])
                    min_lr_box       = gr.Number(label=T["train_min_lr"],
                                                 value=DEFAULT_CONFIG["training"]["min_lr"])

                with gr.Row():
                    step_size_box        = gr.Number(label="Step Size",
                                                     value=DEFAULT_CONFIG["training"]["step_size"])
                    step_gamma_box       = gr.Number(label="Step Gamma",
                                                     value=DEFAULT_CONFIG["training"]["step_gamma"])
                    polynomial_power_box = gr.Number(label="Polynomial Power",
                                                     value=DEFAULT_CONFIG["training"]["polynomial_power"])
                    save_interval_box    = gr.Number(label=T["train_save_interval"],
                                                     value=DEFAULT_CONFIG["training"]["save_interval"])

                train_btn = gr.Button(T["train_start_btn"])
                stop_btn  = gr.Button(T["stop_btn"])

                with gr.Row():
                    with gr.Column(scale=1):
                        train_progress = gr.HTML(label="Training Progress")
                        train_log = gr.HTML(label=T["train_log"], elem_id="train-log-box")
                    with gr.Column(scale=2):
                        train_plot     = gr.Image(label=T["train_plot"], type="pil")

            # -------------- 推理 Tab -------------- #
            with gr.Tab(T["infer_tab"]) as inf_tab:
                with gr.Row():
                    data_dir_inf = gr.Textbox(label=T["dp_processed_dir"], value="", interactive=False)
                    out_dir_inf  = gr.Textbox(label=T["inf_out_dir"], value="", interactive=False)

                prompt_box = gr.Textbox(label=T["inf_prompt"],
                                        value=DEFAULT_CONFIG["inference"]["prompt"], lines=5)

                with gr.Row():
                    num_samples_box    = gr.Number(label=T["inf_num_samples"],
                                                   value=DEFAULT_CONFIG["inference"]["num_samples"])
                    max_new_tokens_box = gr.Number(label=T["inf_max_new_tokens"],
                                                   value=DEFAULT_CONFIG["inference"]["max_new_tokens"])
                    temperature_box    = gr.Number(label=T["inf_temperature"],
                                                   value=DEFAULT_CONFIG["inference"]["temperature"])
                    top_k_box          = gr.Number(label=T["inf_top_k"],
                                                   value=DEFAULT_CONFIG["inference"]["top_k"])
                    seed_box_inf       = gr.Number(label=T["inf_seed"],
                                                   value=DEFAULT_CONFIG["inference"]["seed"])

                inf_btn    = gr.Button(T["inf_start_btn"])
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
                # Get current language
                current_lang = lang_select.value
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
                    num_proc=int(num_proc_)
                )
                new_choices = _model_choices()
                new_val     = f"{info['model_id']} - {model_name.strip() or 'unnamed'}"
                msg = (
                    f"✅ {T_current['dp_result']}:\n"  # Use current langeuage
                    f"model_id = {info['model_id']}\n"
                    f"processed_dir = {info['processed_data_dir']}\n"
                    f"vocab_size = {info['vocab_size']}\n"
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

        # ------------------------------------------------------------------ #
        # Call backs: stop training
        # ------------------------------------------------------------------ #
        stop_btn.click(fn=stop_training, inputs=[], outputs=[])

        # -----------------------------
        # LR Scheduler Callback
        # -----------------------------
        def update_lr_scheduler_params(scheduler_type):
            """
            根据选择的学习率调度器类型，更新相关参数的交互状态和值
            """
            # 初始化所有参数框的状态
            warmup_update = gr.update(interactive=False, value="")
            lr_decay_update = gr.update(interactive=False, value="")
            min_lr_update = gr.update(interactive=False, value="")
            step_size_update = gr.update(interactive=False, value="")
            step_gamma_update = gr.update(interactive=False, value="")
            polynomial_power_update = gr.update(interactive=False, value="")
            
            # 根据调度器类型设置相应参数框的状态
            if scheduler_type == "none":
                # 所有参数都不需要
                pass
            
            elif scheduler_type == "cosine":
                # 余弦调度器需要 warmup_iters, lr_decay_iters, min_lr
                warmup_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["warmup_iters"]
                )
                lr_decay_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["lr_decay_iters"]
                )
                min_lr_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["min_lr"]
                )
                
            elif scheduler_type == "constant_with_warmup":
                # 常数调度器只需要 warmup_iters
                warmup_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["warmup_iters"]
                )
                
            elif scheduler_type == "linear":
                # 线性调度器需要 warmup_iters, lr_decay_iters, min_lr
                warmup_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["warmup_iters"]
                )
                lr_decay_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["lr_decay_iters"]
                )
                min_lr_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["min_lr"]
                )
                
            elif scheduler_type == "step":
                # 步长调度器需要 step_size, step_gamma
                step_size_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["step_size"]
                )
                step_gamma_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["step_gamma"]
                )
                
            elif scheduler_type == "polynomial":
                # 多项式调度器需要所有参数
                warmup_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["warmup_iters"]
                )
                lr_decay_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["lr_decay_iters"]
                )
                min_lr_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["min_lr"]
                )
                polynomial_power_update = gr.update(
                    interactive=True, 
                    value=DEFAULT_CONFIG["training"]["polynomial_power"]
                )
                
            return [
                warmup_update,
                lr_decay_update,
                min_lr_update,
                step_size_update,
                step_gamma_update,
                polynomial_power_update
            ]
        
        # 连接学习率调度器下拉框的change事件到回调函数
        lr_scheduler_box.change(
            fn=update_lr_scheduler_params,
            inputs=[lr_scheduler_box],
            outputs=[
                warmup_box,
                lr_decay_box,
                min_lr_box,
                step_size_box,
                step_gamma_box,
                polynomial_power_box
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
            img_pil = None
            try:
                num_eval_seeds_int = int(num_eval_seeds_)
                if num_eval_seeds_int < 0 or num_eval_seeds_int > 2**32-1:
                    raise ValueError("seed out of range")
            except ValueError as e:
                yield (f"<div style='color:red;'>{str(e)}</div>", str(e), img_pil); return

            try:
                gen = train_model_generator(
                    data_dir=data_dir_,
                    out_dir=out_dir_,
                    plot_interval=int(plot_interval_),
                    log_interval=int(log_interval_),
                    num_eval_seeds=int(num_eval_seeds_),
                    save_best_val_checkpoint=bool(save_best_val_ckpt_),
                    init_from=init_from_,
                    gradient_accumulation_steps=int(grad_acc_),
                    batch_size=int(batch_size_), block_size=int(block_size_),
                    n_layer=int(n_layer_), n_head=int(n_head_), n_embd=int(n_embd_),
                    dropout=float(dropout_), bias=bool(bias_),
                    learning_rate=float(lr_), max_iters=int(max_iters_),
                    weight_decay=float(weight_decay_),
                    beta1=float(beta1_), beta2=float(beta2_),
                    lr_scheduler_type=lr_scheduler_type_,
                    warmup_iters=int(warmup_), lr_decay_iters=int(lr_decay_),
                    min_lr=float(min_lr_), step_size=int(step_size_),
                    step_gamma=float(step_gamma_), polynomial_power=float(polynomial_power_),
                    backend=backend_, device=device_, dtype=dtype_,
                    compile_model=bool(compile_), seed=int(seed_), save_interval=int(save_interval_)
                )
                for p_html, log_html, img in gen:
                    formatted_log = f"<div>{log_html}</div>"
                    yield (p_html, log_html, img)
            except Exception as e:
                err = f"Error: {str(e)}"
                yield (f"<div style='color:red;'>{err}</div>", err, img_pil)

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
                gen = generate_text(
                    data_dir=data_dir_inf_, out_dir=out_dir_inf_,
                    prompt=prompt_,
                    num_samples=int(num_samples_),
                    max_new_tokens=int(max_new_tokens_),
                    temperature=float(temperature_),
                    top_k=int(top_k_) if top_k_ else None,
                    seed=int(seed_inf_),
                    device=DEFAULT_CONFIG["inference"]["device"],
                    dtype=DEFAULT_CONFIG["inference"]["dtype"],
                    compile_model=DEFAULT_CONFIG["inference"]["compile_model"]
                )
                acc = ""
                for piece in gen:
                    acc += piece + "\n\n"
                    yield acc.strip()
            except Exception as e:
                yield f"Error: {str(e)}"

        inf_btn.click(
            fn=inference_cb,
            inputs=[data_dir_inf, out_dir_inf, prompt_box,
                    num_samples_box, max_new_tokens_box,
                    temperature_box, top_k_box, seed_box_inf],
            outputs=inf_output
        )

        # ------------------------------------------------------------------ #
        # Call backs: model selection
        # ------------------------------------------------------------------ #
        def _reset_updates():
            def _d(val=""): return gr.update(value=val)
            d = DEFAULT_CONFIG
            return [
                gr.update(value=True), _d("new_model"),          # new_model_chk & model_name_box
                _d(), _d(),                                      # data_dir_box, out_dir_box
                _d(d["training"]["plot_interval"]), _d(d["training"]["log_interval"]),
                _d(d["training"]["num_eval_seeds"]),
                _d(d["training"]["save_best_val_checkpoint"]),
                _d(d["training"]["init_from"]),
                _d(d["training"]["gradient_accumulation_steps"]),
                _d(d["training"]["batch_size"]),
                _d(d["training"]["block_size"]),
                _d(d["training"]["n_layer"]),
                _d(d["training"]["n_head"]),
                _d(d["training"]["n_embd"]),
                _d(d["training"]["dropout"]),
                _d(d["training"]["bias"]),
                _d(d["training"]["learning_rate"]),
                _d(d["training"]["max_iters"]),
                _d(d["training"]["weight_decay"]),
                _d(d["training"]["beta1"]),
                _d(d["training"]["beta2"]),
                _d(d["training"]["lr_scheduler_type"]),
                _d(d["training"]["warmup_iters"]),
                _d(d["training"]["lr_decay_iters"]),
                _d(d["training"]["min_lr"]),
                _d(d["training"]["step_size"]),
                _d(d["training"]["step_gamma"]),
                _d(d["training"]["polynomial_power"]),
                _d(d["training"]["backend"]),
                _d(d["training"]["device"]),
                _d(d["training"]["dtype"]),
                _d(d["training"]["compile_model"]),
                _d(d["training"]["seed"]),
                _d(d["training"]["save_interval"]),
                None, "",                                     # train_plot, train_log
                _d(), _d(),                                   # data_dir_inf, out_dir_inf
                _d(d["inference"]["prompt"]),
                _d(d["inference"]["num_samples"]),
                _d(d["inference"]["max_new_tokens"]),
                _d(d["inference"]["temperature"]),
                _d(d["inference"]["top_k"]),
                _d(d["inference"]["seed"]),
                ""                                            # inf_output
            ]

        def select_model_cb(sel: str):
            if not sel:
                return _reset_updates()

            try:
                mid = int(sel.split(" - ")[0])
            except ValueError:
                return _reset_updates()

            cfg  = dbm.get_training_config(mid)  or {}
            icfg = dbm.get_inference_config(mid) or {}
            info = dbm.get_model_basic_info(mid) or {}
            name = info.get("name", "")
            folder = f"{name}_{mid}"
            data_processed_dir = os.path.join("data", folder, "processed")
            out_dir_root       = os.path.join("out", folder)

            def _cfg(k, d):  return cfg.get(k, d)
            def _ic(k, d):   return icfg.get(k, d)

            # 加载训练损失日志
            loss_log_path = dbm.get_training_log_path(mid)
            loss_plot = _load_loss_plot(loss_log_path)
            train_log_s = ""
            
            # 从loss_log.pkl中读取训练历史记录并格式化
            if loss_log_path and os.path.exists(loss_log_path):
                try:
                    with open(loss_log_path, 'rb') as f:
                        loss_dict = pickle.load(f)
                        
                    train_steps = loss_dict.get('train_plot_steps', [])
                    train_losses = loss_dict.get('train_plot_losses', [])
                    val_steps = loss_dict.get('val_plot_steps', [])
                    val_losses = loss_dict.get('val_plot_losses', [])
                    
                    # 构建格式化的训练日志
                    log_lines = []
                    if train_steps:
                        log_lines.append(f"Training history for model {mid} - {name}:")
                        
                        # 添加训练损失记录
                        for i, (step, loss) in enumerate(zip(train_steps, train_losses)):
                            log_line = f"Step {step}: train_loss={loss:.4f}"
                            
                            # 如果有对应的验证损失，也添加进来
                            if val_steps and i < len(val_steps) and val_steps[i] == step:
                                log_line += f", val_loss={val_losses[i]:.4f}"
                                
                            log_lines.append(log_line)
                            
                            # 限制日志行数，避免过长
                            if i >= 200:  # 限制最多显示200行
                                log_lines.append(f"... (showing first 200 of {len(train_steps)} records)")
                                break
                                
                    train_log_s = "\n".join(log_lines)
                except Exception as e:
                    train_log_s = f"Error loading training log: {str(e)}"

            updates = [
                gr.update(value=False),     # new_model_chk
                gr.update(value=name),      # model_name_box
                gr.update(value=data_processed_dir),   # data_dir_box
                gr.update(value=out_dir_root),         # out_dir_box
                gr.update(value=_cfg("plot_interval",           DEFAULT_CONFIG["training"]["plot_interval"])),
                gr.update(value=_cfg("log_interval",            DEFAULT_CONFIG["training"]["log_interval"])),
                gr.update(value=_cfg("num_eval_seeds",          DEFAULT_CONFIG["training"]["num_eval_seeds"])),
                gr.update(value=_cfg("save_best_val_checkpoint",DEFAULT_CONFIG["training"]["save_best_val_checkpoint"])),
                gr.update(value=_cfg("init_from",               DEFAULT_CONFIG["training"]["init_from"])),
                gr.update(value=_cfg("gradient_accumulation_steps",
                                     DEFAULT_CONFIG["training"]["gradient_accumulation_steps"])),
                gr.update(value=_cfg("batch_size",      DEFAULT_CONFIG["training"]["batch_size"])),
                gr.update(value=_cfg("block_size",      DEFAULT_CONFIG["training"]["block_size"])),
                gr.update(value=_cfg("n_layer",         DEFAULT_CONFIG["training"]["n_layer"])),
                gr.update(value=_cfg("n_head",          DEFAULT_CONFIG["training"]["n_head"])),
                gr.update(value=_cfg("n_embd",          DEFAULT_CONFIG["training"]["n_embd"])),
                gr.update(value=_cfg("dropout",         DEFAULT_CONFIG["training"]["dropout"])),
                gr.update(value=_cfg("bias",            DEFAULT_CONFIG["training"]["bias"])),
                gr.update(value=_cfg("learning_rate",   DEFAULT_CONFIG["training"]["learning_rate"])),
                gr.update(value=_cfg("max_iters",       DEFAULT_CONFIG["training"]["max_iters"])),
                gr.update(value=_cfg("weight_decay",    DEFAULT_CONFIG["training"]["weight_decay"])),
                gr.update(value=_cfg("beta1",           DEFAULT_CONFIG["training"]["beta1"])),
                gr.update(value=_cfg("beta2",           DEFAULT_CONFIG["training"]["beta2"])),
                gr.update(value=_cfg("lr_scheduler_type", DEFAULT_CONFIG["training"]["lr_scheduler_type"])),
                gr.update(value=_cfg("warmup_iters",    DEFAULT_CONFIG["training"]["warmup_iters"])),
                gr.update(value=_cfg("lr_decay_iters",  DEFAULT_CONFIG["training"]["lr_decay_iters"])),
                gr.update(value=_cfg("min_lr",          DEFAULT_CONFIG["training"]["min_lr"])),
                gr.update(value=_cfg("step_size",       DEFAULT_CONFIG["training"]["step_size"])),
                gr.update(value=_cfg("step_gamma",      DEFAULT_CONFIG["training"]["step_gamma"])),
                gr.update(value=_cfg("polynomial_power",DEFAULT_CONFIG["training"]["polynomial_power"])),
                gr.update(value=_cfg("backend",         DEFAULT_CONFIG["training"]["backend"])),
                gr.update(value=_cfg("device",          DEFAULT_CONFIG["training"]["device"])),
                gr.update(value=_cfg("dtype",           DEFAULT_CONFIG["training"]["dtype"])),
                gr.update(value=_cfg("compile_model",   DEFAULT_CONFIG["training"]["compile_model"])),
                gr.update(value=_cfg("seed",            DEFAULT_CONFIG["training"]["seed"])),
                gr.update(value=_cfg("save_interval",   DEFAULT_CONFIG["training"]["save_interval"])),
                loss_plot, train_log_s,
                gr.update(value=data_processed_dir),    # data_dir_inf
                gr.update(value=out_dir_root),          # out_dir_inf
                gr.update(value=_ic("prompt",         DEFAULT_CONFIG["inference"]["prompt"])),
                gr.update(value=_ic("num_samples",    DEFAULT_CONFIG["inference"]["num_samples"])),
                gr.update(value=_ic("max_new_tokens", DEFAULT_CONFIG["inference"]["max_new_tokens"])),
                gr.update(value=_ic("temperature",    DEFAULT_CONFIG["inference"]["temperature"])),
                gr.update(value=_ic("top_k",          DEFAULT_CONFIG["inference"]["top_k"])),
                gr.update(value=_ic("seed",           DEFAULT_CONFIG["inference"]["seed"])),
                dbm.get_inference_history(mid) or ""
            ]
            return updates

        model_dropdown.change(
            fn=select_model_cb,
            inputs=[model_dropdown],
            outputs=[
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
        )

        # ------------------------------------------------------------------ #
        # Call backs: delete model
        # ------------------------------------------------------------------ #
        def delete_model_cb(sel: str):
            if sel and " - " in sel:
                try:
                    dbm.delete_model(int(sel.split(" - ")[0]))
                except Exception:
                    pass
                    
            new_choices = _model_choices()
            return gr.update(choices=new_choices, value=None), *_reset_updates()

        delete_model_btn.click(
            fn=delete_model_cb,
            inputs=[model_dropdown],
            outputs=[model_dropdown,
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
                     inf_output]
        )

        refresh_models_btn.click(lambda: gr.update(choices=_model_choices()), [], [model_dropdown])

        # ------------------------------------------------------------------ #
        # Call backs: language switch
        # ------------------------------------------------------------------ #
        def switch_language(lang_code: str):
            Tn = LANG_JSON[lang_code]
            return [
                gr.update(label=Tn["language_label"], value=lang_code),
                # Tab labels
                gr.update(label=Tn["data_process_tab"]),
                gr.update(label=Tn["train_tab"]),
                gr.update(label=Tn["infer_tab"]),
        
                # Top bar
                gr.update(label=Tn["registered_models"]),  # model_dropdown
                gr.update(value=Tn["refresh_tables"]),     # Button: refresh_models_btn
                gr.update(value=Tn["delete_selected_model"]), # Button: delete_model_btn
        
                # Model management panel
                gr.update(label=Tn["new_model"]),        # new_model_chk
                gr.update(label=Tn["model_name"]),       # model_name_box
        
                # Data processing panel
                gr.update(label=Tn["dp_paste_text"]),      # input_text
                gr.update(label=Tn["dp_txt_dir"]),         # txt_dir
                gr.update(label=Tn["dp_no_val_set"]),      # no_val_set
                gr.update(label=Tn["dp_use_gpt2_tokenizer"]), # use_gpt2
                gr.update(label=Tn["dp_train_split"]),     # train_split
                gr.update(label=Tn["dp_num_proc"]),        # num_proc
                gr.update(value=Tn["dp_start_btn"]),       # process_btn (Button)
                gr.update(label=Tn["dp_result"]),          # process_output
                
                # Training panel
                gr.update(value=f"### {Tn['train_params_title']}"), # train_params_title_md
                gr.update(label=Tn["train_data_dir"]),     # data_dir_box
                gr.update(label=Tn["train_out_dir"]),      # out_dir_box
                gr.update(label=Tn["train_backend"]),      # backend_box
                gr.update(label=Tn["train_device"]),       # device_box
                gr.update(label=Tn["train_dtype"]),        # dtype_box
                gr.update(label=Tn["train_compile_model"]), # compile_box
                gr.update(label=Tn["train_eval_interval"]), # plot_interval_box
                gr.update(label=Tn["train_log_interval"]), # log_interval_box
                gr.update(label=Tn["train_num_eval_seeds"]), # num_eval_seeds_box
                gr.update(label=Tn["train_save_best_val_ckpt"]), # save_best_val_ckpt_box
                gr.update(label=Tn["train_init_from"]),    # init_from_box
                gr.update(label=Tn["train_seed"]),         # seed_box
                gr.update(label=Tn["train_gas"]),          # grad_acc_box
                gr.update(label=Tn["train_batch_size"]),   # batch_size_box
                gr.update(label=Tn["train_block_size"]),   # block_size_box
                gr.update(label=Tn["train_n_layer"]),      # n_layer_box
                gr.update(label=Tn["train_n_head"]),       # n_head_box
                gr.update(label=Tn["train_n_embd"]),       # n_embd_box
                gr.update(label=Tn["train_dropout"]),      # dropout_box
                gr.update(label=Tn["train_bias"]),         # bias_box
                gr.update(label=Tn["train_lr"]),           # lr_box
                gr.update(label=Tn["train_max_iters"]),    # max_iters_box
                gr.update(label=Tn["train_weight_decay"]), # weight_decay_box
                gr.update(label=Tn["train_beta1"]),        # beta1_box
                gr.update(label=Tn["train_beta2"]),        # beta2_box
                gr.update(label=Tn["train_lr_scheduler"]), # lr_scheduler_box
                gr.update(label=Tn["train_warmup_iters"]), # warmup_box
                gr.update(label=Tn["train_lr_decay_iters"]), # lr_decay_box
                gr.update(label=Tn["train_min_lr"]),       # min_lr_box
                gr.update(label=Tn["train_save_interval"]), # save_interval_box
                gr.update(value=Tn["train_start_btn"]),    # train_btn (Button)
                gr.update(value=Tn["stop_btn"]),           # stop_btn (Button)
                gr.update(label=Tn["train_log"]),          # train_log
                gr.update(label=Tn["train_plot"]),         # train_plot
                
                # Inference panel
                gr.update(label=Tn["dp_processed_dir"]),   # data_dir_inf
                gr.update(label=Tn["inf_out_dir"]),        # out_dir_inf
                gr.update(label=Tn["inf_prompt"]),         # prompt_box
                gr.update(label=Tn["inf_num_samples"]),    # num_samples_box
                gr.update(label=Tn["inf_max_new_tokens"]), # max_new_tokens_box
                gr.update(label=Tn["inf_temperature"]),    # temperature_box
                gr.update(label=Tn["inf_top_k"]),          # top_k_box
                gr.update(value=Tn["inf_start_btn"]),      # inf_btn (Button)
                gr.update(label=Tn["inf_result"]),         # inf_output
                gr.update(label=Tn["inf_seed"])            # seed_box_inf
            ]

        lang_select.change(
            fn=switch_language,
            inputs=[lang_select],
            outputs=[
                lang_select,
                data_process_tab, train_tab, inf_tab,
                model_dropdown, refresh_models_btn, delete_model_btn,
                new_model_chk, model_name_box,
                input_text, txt_dir,
                no_val_set, use_gpt2,
                train_split, num_proc, process_btn, process_output,
                train_params_title_md,
                data_dir_box, out_dir_box,
                backend_box, device_box, dtype_box, compile_box,
                plot_interval_box, log_interval_box, num_eval_seeds_box,
                save_best_val_ckpt_box, init_from_box, seed_box,
                grad_acc_box, batch_size_box, block_size_box,
                n_layer_box, n_head_box, n_embd_box,
                dropout_box, bias_box,
                lr_box, max_iters_box, weight_decay_box,
                beta1_box, beta2_box, lr_scheduler_box,
                warmup_box, lr_decay_box, min_lr_box,
                save_interval_box, train_btn, stop_btn,
                train_log, train_plot,
                data_dir_inf, out_dir_inf, prompt_box,
                num_samples_box, max_new_tokens_box,
                temperature_box, top_k_box, inf_btn, inf_output,
                seed_box_inf
            ]
        )

        # 初始化UI时，触发一次学习率调度器相关参数的更新
        # 使用默认值进行一次初始化
        default_scheduler = DEFAULT_CONFIG["training"]["lr_scheduler_type"]
        
        # 不要直接使用update方法，而是通过一个dummy event处理初始化
        # 这样gradio会在界面加载时正确应用这些更新
        demo.load(
            fn=lambda: update_lr_scheduler_params(default_scheduler),
            inputs=None,
            outputs=[
                warmup_box,
                lr_decay_box,
                min_lr_box,
                step_size_box,
                step_gamma_box,
                polynomial_power_box
            ]
        )

    return demo

# ----------------- Launch -------------------
if __name__=="__main__":
    demo = build_app_interface()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
