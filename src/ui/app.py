import torch
import torch._dynamo

import gradio as gr

from src.config import DEFAULT_CONFIG, LANG_JSON
from src.device_manager import device_manager
from src.ui.bindings.comparison import bind_comparison
from src.ui.bindings.data_processing import bind_data_processing
from src.ui.bindings.inference import bind_inference
from src.ui.bindings.language import bind_language
from src.ui.bindings.model_management import bind_model_management
from src.ui.bindings.sft import bind_sft
from src.ui.bindings.training import bind_training
from src.ui.constants import CUSTOM_CSS
from src.ui.helpers import _get_model_choices_list, _match_device_value

torch._dynamo.config.suppress_errors = True


def build_app_interface(selected_lang: str = "zh"):
    """
    Top-level UI function
    Implemented:
        · Logic for new model/model name, automatic directory, dropdown refresh/delete
        · Language switching: After switching the `lang_select` dropdown, **all component labels & default values** are refreshed synchronously
        · Dynamic HTML/SVG loss plot
    """

    T = LANG_JSON[selected_lang]

    with gr.Blocks(title=T["app_title"], css=CUSTOM_CSS) as demo:

        # ========= Top: language ========= #
        lang_select = gr.Dropdown(
            label=T["language_label"],
            choices=list(LANG_JSON.keys()),
            value=selected_lang,
            interactive=True,
        )

        # ========= Model management ========= #
        with gr.Row(equal_height=True):
            model_dropdown = gr.Dropdown(label=T["registered_models"], choices=_get_model_choices_list(), value=None, interactive=True, scale=2)
            with gr.Column(scale=1):
                refresh_models_btn = gr.Button(T["refresh_tables"])
                delete_model_btn = gr.Button(T["delete_selected_model"], variant="stop")

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
                            no_val_set = gr.Checkbox(
                                label=T["dp_no_val_set"],
                                value=DEFAULT_CONFIG["data_process"]["no_validation"],
                            )
                            use_gpt2 = gr.Checkbox(
                                label=T["dp_use_gpt2_tokenizer"],
                                value=DEFAULT_CONFIG["data_process"]["use_gpt2_tokenizer"],
                            )
                        train_split = gr.Slider(
                            label=T["dp_train_split"],
                            minimum=0.1,
                            maximum=0.99,
                            step=0.01,
                            value=DEFAULT_CONFIG["data_process"]["train_split_ratio"],
                        )
                        num_proc = gr.Number(
                            label=T["dp_num_proc"],
                            value=DEFAULT_CONFIG["data_process"]["num_proc"],
                            precision=0,
                        )
                process_btn = gr.Button(T["dp_start_btn"])
                process_output = gr.Textbox(label=T["dp_result"], lines=5, interactive=False)

            # -------------- Training Tab -------------- #
            with gr.Tab(T["train_tab"]) as train_tab:
                train_params_title_md = gr.Markdown(f"### {T['train_params_title']}")

                with gr.Row():
                    data_dir_box = gr.Textbox(label=T["train_data_dir"], value="", interactive=False)
                    out_dir_box = gr.Textbox(label=T["train_out_dir"], value="", interactive=False)
                    backend_box = gr.Dropdown(
                        label=T["train_backend"],
                        choices=["nccl", "gloo"],
                        value=DEFAULT_CONFIG["training"]["backend"],
                    )
                    available_devices = device_manager.get_available_devices_list()
                    device_box = gr.Dropdown(
                        label=T["train_device"],
                        choices=available_devices,
                        value=_match_device_value(DEFAULT_CONFIG["training"]["device"], available_devices),
                    )
                    dtype_box = gr.Dropdown(
                        label=T["train_dtype"],
                        choices=["float16", "bfloat16", "float32"],
                        value=DEFAULT_CONFIG["training"]["dtype"],
                    )
                    compile_box = gr.Checkbox(
                        label=T["train_compile_model"],
                        value=DEFAULT_CONFIG["training"]["compile_model"],
                    )

                with gr.Row():
                    plot_interval_box = gr.Number(
                        label=T["train_eval_interval"],
                        value=DEFAULT_CONFIG["training"]["plot_interval"],
                    )
                    log_interval_box = gr.Number(
                        label=T["train_log_interval"],
                        value=DEFAULT_CONFIG["training"]["log_interval"],
                    )
                    num_eval_seeds_box = gr.Number(
                        label=T["train_num_eval_seeds"],
                        value=DEFAULT_CONFIG["training"]["num_eval_seeds"],
                    )
                    save_best_val_ckpt_box = gr.Checkbox(
                        label=T["train_save_best_val_ckpt"],
                        value=DEFAULT_CONFIG["training"]["save_best_val_checkpoint"],
                    )
                    init_from_box = gr.Dropdown(
                        label=T["train_init_from"],
                        choices=["scratch", "resume"],
                        value=DEFAULT_CONFIG["training"]["init_from"],
                    )
                    seed_box = gr.Number(
                        label=T["train_seed"],
                        value=DEFAULT_CONFIG["training"]["seed"],
                    )

                with gr.Row():
                    grad_acc_box = gr.Number(
                        label=T["train_gas"],
                        value=DEFAULT_CONFIG["training"]["gradient_accumulation_steps"],
                    )
                    batch_size_box = gr.Number(
                        label=T["train_batch_size"],
                        value=DEFAULT_CONFIG["training"]["batch_size"],
                    )
                    block_size_box = gr.Number(
                        label=T["train_block_size"],
                        value=DEFAULT_CONFIG["training"]["block_size"],
                    )
                    n_layer_box = gr.Number(
                        label=T["train_n_layer"],
                        value=DEFAULT_CONFIG["training"]["n_layer"],
                    )
                    n_head_box = gr.Number(
                        label=T["train_n_head"],
                        value=DEFAULT_CONFIG["training"]["n_head"],
                    )
                    n_embd_box = gr.Number(
                        label=T["train_n_embd"],
                        value=DEFAULT_CONFIG["training"]["n_embd"],
                    )

                with gr.Row():
                    dropout_box = gr.Number(
                        label=T["train_dropout"],
                        value=DEFAULT_CONFIG["training"]["dropout"],
                    )
                    bias_box = gr.Checkbox(
                        label=T["train_bias"],
                        value=DEFAULT_CONFIG["training"]["bias"],
                    )
                    lr_box = gr.Number(
                        label=T["train_lr"],
                        value=DEFAULT_CONFIG["training"]["learning_rate"],
                    )
                    max_iters_box = gr.Number(
                        label=T["train_max_iters"],
                        value=DEFAULT_CONFIG["training"]["max_iters"],
                    )
                    weight_decay_box = gr.Number(
                        label=T["train_weight_decay"],
                        value=DEFAULT_CONFIG["training"]["weight_decay"],
                    )

                with gr.Row():
                    beta1_box = gr.Number(
                        label=T["train_beta1"],
                        value=DEFAULT_CONFIG["training"]["beta1"],
                    )
                    beta2_box = gr.Number(
                        label=T["train_beta2"],
                        value=DEFAULT_CONFIG["training"]["beta2"],
                    )
                    lr_scheduler_box = gr.Dropdown(
                        label=T["train_lr_scheduler"],
                        choices=[
                            "none",
                            "cosine",
                            "constant_with_warmup",
                            "linear",
                            "step",
                            "polynomial",
                        ],
                        value=DEFAULT_CONFIG["training"]["lr_scheduler_type"],
                    )
                    warmup_box = gr.Number(
                        label=T["train_warmup_iters"],
                        value=DEFAULT_CONFIG["training"]["warmup_iters"],
                    )
                    lr_decay_box = gr.Number(
                        label=T["train_lr_decay_iters"],
                        value=DEFAULT_CONFIG["training"]["lr_decay_iters"],
                    )
                    min_lr_box = gr.Number(
                        label=T["train_min_lr"],
                        value=DEFAULT_CONFIG["training"]["min_lr"],
                    )

                with gr.Row():
                    step_size_box = gr.Number(
                        label="Step Size",
                        value=DEFAULT_CONFIG["training"]["step_size"],
                    )
                    step_gamma_box = gr.Number(
                        label="Step Gamma",
                        value=DEFAULT_CONFIG["training"]["step_gamma"],
                    )
                    polynomial_power_box = gr.Number(
                        label="Polynomial Power",
                        value=DEFAULT_CONFIG["training"]["polynomial_power"],
                    )
                    save_interval_box = gr.Number(
                        label=T["train_save_interval"],
                        value=DEFAULT_CONFIG["training"]["save_interval"],
                    )

                # Self-attention parameters in collapsible accordion
                with gr.Accordion(label=T["train_self_attn_title"], open=False) as self_attn_accordion:
                    use_self_attention_box = gr.Checkbox(
                        label=T["train_use_self_attention"],
                        value=DEFAULT_CONFIG["training"]["use_self_attention"],
                    )

                    with gr.Row():
                        ffn_hidden_mult_box = gr.Number(
                            label=T["train_ffn_hidden_mult"],
                            value=DEFAULT_CONFIG["training"]["ffn_hidden_mult"],
                            visible=False,
                        )
                        qkv_bias_box = gr.Checkbox(
                            label=T["train_qkv_bias"],
                            value=DEFAULT_CONFIG["training"]["qkv_bias"],
                            visible=False,
                        )
                        attn_dropout_box = gr.Number(
                            label=T["train_attn_dropout"],
                            value=DEFAULT_CONFIG["training"]["attn_dropout"],
                            step=0.01,
                            visible=False,
                        )
                        resid_dropout_box = gr.Number(
                            label=T["train_resid_dropout"],
                            value=DEFAULT_CONFIG["training"]["resid_dropout"],
                            step=0.01,
                            visible=False,
                        )

                    with gr.Row():
                        ln_eps_box = gr.Number(
                            label=T["train_ln_eps"],
                            value=DEFAULT_CONFIG["training"]["ln_eps"],
                            step=1e-6,
                            visible=False,
                        )
                        init_std_box = gr.Number(
                            label=T["train_init_std"],
                            value=DEFAULT_CONFIG["training"]["init_std"],
                            step=0.001,
                            visible=False,
                        )
                        use_flash_attn_box = gr.Checkbox(
                            label=T["train_use_flash_attn"],
                            value=DEFAULT_CONFIG["training"]["use_flash_attn"],
                            visible=False,
                        )
                        pos_encoding_type_box = gr.Dropdown(
                            label=T["train_pos_encoding_type"],
                            choices=["rope", "alibi"],
                            value=DEFAULT_CONFIG["training"]["pos_encoding_type"],
                            visible=False,
                        )

                    # New optimized parameters - row 1
                    with gr.Row():
                        rope_base_box = gr.Number(
                            label=T["train_rope_base"],
                            value=DEFAULT_CONFIG["training"]["rope_base"],
                            visible=False,
                        )
                        rope_cache_size_box = gr.Number(
                            label=T["train_rope_cache_size"],
                            value=DEFAULT_CONFIG["training"]["rope_cache_size"],
                            visible=False,
                            info="Cache size for RoPE (0 for auto)",
                        )
                        alibi_bias_scale_box = gr.Number(
                            label=T["train_alibi_bias_scale"],
                            value=DEFAULT_CONFIG["training"]["alibi_bias_scale"],
                            step=0.1,
                            visible=False,
                            info="Scaling factor for ALiBi bias",
                        )
                        ffn_activation_box = gr.Dropdown(
                            label=T["train_ffn_activation"],
                            choices=["gelu", "relu", "swish"],
                            value=DEFAULT_CONFIG["training"]["ffn_activation"],
                            visible=False,
                            info="FFN activation function",
                        )

                    # New optimized parameters - row 2
                    with gr.Row():
                        attention_scale_factor_box = gr.Number(
                            label=T["train_attention_scale_factor"],
                            value=DEFAULT_CONFIG["training"]["attention_scale_factor"],
                            step=0.1,
                            visible=False,
                            info="Additional attention scaling",
                        )
                        gradient_checkpointing_box = gr.Checkbox(
                            label=T["train_gradient_checkpointing"],
                            value=DEFAULT_CONFIG["training"]["gradient_checkpointing"],
                            visible=False,
                            info="Enable gradient checkpointing to save memory",
                        )
                        cache_strategy_box = gr.Dropdown(
                            label=T["train_cache_strategy"],
                            choices=["adaptive", "fixed", "minimal"],
                            value=DEFAULT_CONFIG["training"]["cache_strategy"],
                            visible=False,
                            info="Cache allocation strategy",
                        )
                        max_cache_size_box = gr.Number(
                            label=T["train_max_cache_size"],
                            value=DEFAULT_CONFIG["training"]["max_cache_size"],
                            visible=False,
                            info="Maximum cache size for dynamic allocation",
                        )

                    # Error handling parameters
                    with gr.Row():
                        strict_validation_box = gr.Checkbox(
                            label=T["train_strict_validation"],
                            value=DEFAULT_CONFIG["training"]["strict_validation"],
                            visible=False,
                            info="Enable strict input validation",
                        )
                        fallback_on_error_box = gr.Checkbox(
                            label=T["train_fallback_on_error"],
                            value=DEFAULT_CONFIG["training"]["fallback_on_error"],
                            visible=False,
                            info="Fallback to basic implementations on error",
                        )

                with gr.Row():
                    train_btn = gr.Button(T["train_start_btn"])
                    stop_btn = gr.Button(T["stop_btn"])

                with gr.Row():
                    with gr.Column(scale=1):
                        train_progress = gr.HTML(label="Training Progress")
                        train_log = gr.HTML(label=T["train_log"], elem_id="train-log-box")
                    with gr.Column(scale=2):
                        train_plot = gr.HTML(label=T["train_plot"])

            # -------------- SFT Tab -------------- #
            with gr.Tab(T["sft_tab"]) as sft_tab:
                sft_title_md = gr.Markdown(f"### {T['sft_title']}")

                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        sft_dataset_example = gr.Code(
                            label=T["sft_dataset_example"],
                            value=T["sft_dataset_example_json"],
                            language="json",
                            interactive=False,
                        )

                    with gr.Column(scale=2):
                        sft_dataset_file = gr.File(
                            label=T["sft_dataset_file"],
                            file_types=[".json"],
                            type="filepath",
                        )
                        sft_dataset_dir = gr.Textbox(
                            label=T["sft_dataset_dir"],
                            placeholder="Or enter directory path containing JSON files...",
                        )

                with gr.Row():
                    sft_format_status = gr.Textbox(
                        label=T["sft_format_status"],
                        value=T["sft_no_dataset"],
                        interactive=False,
                        scale=4,
                    )
                    with gr.Column(scale=1):
                        gr.HTML("<div style='flex-grow: 1;'></div>")
                        sft_validate_btn = gr.Button(T["sft_validate_btn"])

                # Store loaded dataset in state
                sft_dataset_state = gr.State(value=[])

                with gr.Row():
                    sft_init_from = gr.Dropdown(
                        label=T["sft_init_from"],
                        choices=["scratch", "resume"],
                        value=DEFAULT_CONFIG["sft"]["init_from"],
                    )
                    sft_save_best_loss_ckpt = gr.Checkbox(
                        label=T["sft_save_best_loss_ckpt"],
                        value=DEFAULT_CONFIG["sft"]["save_best_loss_checkpoint"],
                    )

                with gr.Accordion(T["sft_basic_params"], open=True) as sft_basic_params_accordion:
                    with gr.Row():
                        with gr.Column():
                            sft_epochs = gr.Number(
                                label=T["sft_epochs"],
                                value=DEFAULT_CONFIG["sft"]["epochs"],
                                minimum=1,
                                maximum=100,
                            )
                            sft_batch_size = gr.Number(
                                label=T["sft_batch_size"],
                                value=DEFAULT_CONFIG["sft"]["batch_size"],
                                minimum=1,
                                maximum=64,
                            )
                            sft_gradient_accumulation = gr.Number(
                                label=T["sft_gradient_accumulation"],
                                value=DEFAULT_CONFIG["sft"]["gradient_accumulation_steps"],
                                minimum=1,
                                maximum=32,
                            )
                        with gr.Column():
                            sft_learning_rate = gr.Number(
                                label=T["sft_learning_rate"],
                                value=DEFAULT_CONFIG["sft"]["learning_rate"],
                                step=1e-6,
                            )
                            sft_lr_scheduler = gr.Dropdown(
                                label=T["sft_lr_scheduler"],
                                choices=[
                                    "none",
                                    "cosine",
                                    "constant_with_warmup",
                                    "linear",
                                    "step",
                                    "polynomial",
                                ],
                                value=DEFAULT_CONFIG["sft"]["lr_scheduler_type"],
                            )
                            sft_max_seq_length = gr.Number(
                                label=T["sft_max_seq_length"],
                                value=DEFAULT_CONFIG["sft"]["max_seq_length"],
                                minimum=32,
                                maximum=4096,
                            )

                with gr.Accordion(T["sft_scheduler_params"], open=False) as sft_scheduler_accordion:
                    with gr.Row():
                        sft_warmup_iters = gr.Number(
                            label=T["sft_warmup_iters"],
                            value=DEFAULT_CONFIG["sft"]["warmup_iters"],
                            minimum=0,
                        )
                        sft_lr_decay_iters = gr.Number(
                            label=T["sft_lr_decay_iters"],
                            value=DEFAULT_CONFIG["sft"]["lr_decay_iters"],
                            minimum=0,
                        )
                        sft_min_lr = gr.Number(
                            label=T["sft_min_lr"],
                            value=DEFAULT_CONFIG["sft"]["min_lr"],
                            step=1e-6,
                            minimum=0,
                        )
                    with gr.Row():
                        sft_step_size = gr.Number(
                            label=T["sft_step_size"],
                            value=DEFAULT_CONFIG["sft"]["step_size"],
                            minimum=0,
                        )
                        sft_step_gamma = gr.Number(
                            label=T["sft_step_gamma"],
                            value=DEFAULT_CONFIG["sft"]["step_gamma"],
                            step=0.01,
                            minimum=0,
                        )
                        sft_polynomial_power = gr.Number(
                            label=T["sft_poly_power"],
                            value=DEFAULT_CONFIG["sft"]["polynomial_power"],
                            step=0.1,
                            minimum=0,
                        )

                with gr.Accordion(T["sft_optim_params"], open=True) as sft_optim_params_accordion:
                    with gr.Row():
                        with gr.Column():
                            sft_weight_decay = gr.Number(
                                label=T["sft_weight_decay"],
                                value=DEFAULT_CONFIG["sft"]["weight_decay"],
                                step=1e-4,
                                minimum=0,
                            )
                            sft_label_smoothing = gr.Number(
                                label=T["sft_label_smoothing"],
                                value=DEFAULT_CONFIG["sft"]["label_smoothing"],
                                step=0.01,
                                minimum=0,
                                maximum=0.9,
                            )
                        with gr.Column():
                            sft_grad_clip = gr.Number(
                                label=T["sft_grad_clip"],
                                value=DEFAULT_CONFIG["sft"]["grad_clip"],
                                step=0.1,
                                minimum=0,
                            )
                            sft_freeze_layers = gr.Number(
                                label=T["sft_freeze_layers"],
                                value=DEFAULT_CONFIG["sft"]["freeze_layers"],
                                minimum=0,
                                maximum=64,
                            )

                sft_system_prompt = gr.Textbox(
                    label=T["sft_system_prompt"],
                    value=DEFAULT_CONFIG["sft"]["system_prompt"],
                    lines=2,
                )

                with gr.Row():
                    sft_start_btn = gr.Button(T["sft_start_btn"], variant="primary", interactive=False)
                    sft_stop_btn = gr.Button(T["sft_stop_btn"], variant="stop")

                with gr.Row():
                    with gr.Column(scale=1):
                        sft_progress = gr.HTML(label=T["sft_progress"])
                        sft_log = gr.HTML(label=T["sft_log"], elem_id="sft-log-box")
                    with gr.Column(scale=2):
                        sft_plot = gr.HTML(label=T["sft_plot"])

            # -------------- Inference Tab -------------- #
            with gr.Tab(T["infer_tab"]) as inf_tab:
                with gr.Row():
                    data_dir_inf = gr.Textbox(label=T["dp_processed_dir"], value="", interactive=False)
                    out_dir_inf = gr.Textbox(label=T["inf_out_dir"], value="", interactive=False)

                prompt_box = gr.Textbox(
                    label=T["inf_prompt"],
                    value=DEFAULT_CONFIG["inference"]["prompt"],
                    lines=5,
                    placeholder="Just write something...",
                )

                with gr.Row():
                    num_samples_box = gr.Number(
                        label=T["inf_num_samples"],
                        value=DEFAULT_CONFIG["inference"]["num_samples"],
                    )
                    max_new_tokens_box = gr.Number(
                        label=T["inf_max_new_tokens"],
                        value=DEFAULT_CONFIG["inference"]["max_new_tokens"],
                    )
                    temperature_box = gr.Number(
                        label=T["inf_temperature"],
                        value=DEFAULT_CONFIG["inference"]["temperature"],
                        step=0.1,
                    )
                    top_k_box = gr.Number(
                        label=T["inf_top_k"],
                        value=DEFAULT_CONFIG["inference"]["top_k"],
                    )
                    dtype_box_inf = gr.Dropdown(
                        label=T["inf_dtype"],
                        choices=["float16", "bfloat16", "float32"],
                        value=DEFAULT_CONFIG["inference"]["dtype"],
                    )
                    available_devices_inf = device_manager.get_available_devices_list()
                    device_box_inf = gr.Dropdown(
                        label=T["inf_device"],
                        choices=available_devices_inf,
                        value=_match_device_value(DEFAULT_CONFIG["inference"]["device"], available_devices_inf),
                    )
                    seed_box_inf = gr.Number(
                        label=T["inf_seed"],
                        value=DEFAULT_CONFIG["inference"]["seed"],
                    )

                # Chat Mode Toggle
                inf_chat_mode = gr.Checkbox(label=T["inf_chat_mode"], value=False)

                # Chat Interface (Hidden by default)
                with gr.Group(visible=False) as chat_interface_group:
                    chatbot = gr.Chatbot(
                        label=T["inf_chat_history"],
                        height=400,
                        sanitize_html=False,
                        type="messages",
                    )
                    inf_system_prompt = gr.Textbox(
                        label=T["inf_system_prompt"],
                        value=DEFAULT_CONFIG["sft"]["system_prompt"],
                    )
                    with gr.Row():
                        inf_user_input = gr.Textbox(
                            label=T["inf_user_input"],
                            placeholder="Type a message...",
                            scale=4,
                        )
                        with gr.Column(scale=1, min_width=100):
                            inf_send_btn = gr.Button(T["inf_send_btn"], variant="primary")
                            inf_clear_btn = gr.Button(T["inf_clear_chat"])

                    # Chat advanced output section (collapsed by default)
                    with gr.Accordion(T["inf_chat_advanced"], open=False) as chat_advanced_accordion:
                        chat_advanced_output = gr.HTML(label=T["inf_advanced_output"], elem_id="chat-advanced-html")

                # Standard Inference Interface (Hidden when Chat Mode is enabled)
                with gr.Group(visible=True) as standard_interface_group:
                    inf_btn = gr.Button(T["inf_start_btn"])
                    inf_output = gr.HTML(label=T["inf_result"], elem_id="inf-result-html")

                    # Advanced output section (collapsed by default)
                    with gr.Accordion(T["inf_advanced_output"], open=False) as advanced_accordion:
                        inf_advanced_output = gr.HTML(label=T["inf_advanced_output"], elem_id="inf-advanced-html")

            # -------------- Comparison Tab -------------- #
            with gr.Tab(T["compare_tab"]) as comp_tab:
                # Two-column layout for model comparison
                with gr.Row():
                    with gr.Column():
                        comp_left_model = gr.Dropdown(
                            label=T["compare_left_model"],
                            choices=_get_model_choices_list(),
                            value=None,
                            interactive=True,
                        )
                    with gr.Column():
                        comp_right_model = gr.Dropdown(
                            label=T["compare_right_model"],
                            choices=_get_model_choices_list(),
                            value=None,
                            interactive=True,
                        )

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

                # Parameters for left and right models
                with gr.Row():
                    # Left model params
                    with gr.Column():
                        gr.Markdown("**⚙️ Model 1**")
                        with gr.Row():
                            comp_left_num_samples = gr.Number(
                                label=T["inf_num_samples"],
                                value=DEFAULT_CONFIG["inference"]["num_samples"],
                            )
                            comp_left_max_tokens = gr.Number(
                                label=T["inf_max_new_tokens"],
                                value=DEFAULT_CONFIG["inference"]["max_new_tokens"],
                            )
                            comp_left_temperature = gr.Number(
                                label=T["inf_temperature"],
                                value=DEFAULT_CONFIG["inference"]["temperature"],
                                step=0.1,
                            )
                        with gr.Row():
                            comp_left_top_k = gr.Number(
                                label=T["inf_top_k"],
                                value=DEFAULT_CONFIG["inference"]["top_k"],
                            )
                            comp_left_dtype = gr.Dropdown(
                                label=T["inf_dtype"],
                                choices=["float16", "bfloat16", "float32"],
                                value=DEFAULT_CONFIG["inference"]["dtype"],
                            )
                            comp_left_seed = gr.Number(
                                label=T["inf_seed"],
                                value=DEFAULT_CONFIG["inference"]["seed"],
                            )

                    # Right model params
                    with gr.Column():
                        gr.Markdown("**⚙️ Model 2**")
                        with gr.Row():
                            comp_right_num_samples = gr.Number(
                                label=T["inf_num_samples"],
                                value=DEFAULT_CONFIG["inference"]["num_samples"],
                            )
                            comp_right_max_tokens = gr.Number(
                                label=T["inf_max_new_tokens"],
                                value=DEFAULT_CONFIG["inference"]["max_new_tokens"],
                            )
                            comp_right_temperature = gr.Number(
                                label=T["inf_temperature"],
                                value=DEFAULT_CONFIG["inference"]["temperature"],
                                step=0.1,
                            )
                        with gr.Row():
                            comp_right_top_k = gr.Number(
                                label=T["inf_top_k"],
                                value=DEFAULT_CONFIG["inference"]["top_k"],
                            )
                            comp_right_dtype = gr.Dropdown(
                                label=T["inf_dtype"],
                                choices=["float16", "bfloat16", "float32"],
                                value=DEFAULT_CONFIG["inference"]["dtype"],
                            )
                            comp_right_seed = gr.Number(
                                label=T["inf_seed"],
                                value=DEFAULT_CONFIG["inference"]["seed"],
                            )

                # Shared prompt
                comp_prompt = gr.Textbox(
                    label=T["compare_shared_prompt"],
                    lines=5,
                    value=DEFAULT_CONFIG["inference"]["prompt"],
                    placeholder="Just write something...",
                )

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
        bind_data_processing(
            process_btn,
            new_model_chk,
            model_name_box,
            model_dropdown,
            input_text,
            txt_dir,
            train_split,
            no_val_set,
            use_gpt2,
            num_proc,
            lang_select,
            process_output,
            comp_left_model,
            comp_right_model,
        )

        bind_training(
            demo,
            stop_btn,
            lr_scheduler_box,
            warmup_box,
            lr_decay_box,
            min_lr_box,
            step_size_box,
            step_gamma_box,
            polynomial_power_box,
            use_self_attention_box,
            ffn_hidden_mult_box,
            qkv_bias_box,
            attn_dropout_box,
            resid_dropout_box,
            ln_eps_box,
            init_std_box,
            use_flash_attn_box,
            pos_encoding_type_box,
            rope_base_box,
            rope_cache_size_box,
            alibi_bias_scale_box,
            ffn_activation_box,
            attention_scale_factor_box,
            gradient_checkpointing_box,
            cache_strategy_box,
            max_cache_size_box,
            strict_validation_box,
            fallback_on_error_box,
            train_btn,
            data_dir_box,
            out_dir_box,
            plot_interval_box,
            log_interval_box,
            num_eval_seeds_box,
            save_best_val_ckpt_box,
            init_from_box,
            grad_acc_box,
            batch_size_box,
            block_size_box,
            n_layer_box,
            n_head_box,
            n_embd_box,
            dropout_box,
            bias_box,
            lr_box,
            max_iters_box,
            weight_decay_box,
            beta1_box,
            beta2_box,
            backend_box,
            device_box,
            dtype_box,
            compile_box,
            seed_box,
            save_interval_box,
            train_progress,
            train_log,
            train_plot,
        )

        bind_inference(
            inf_chat_mode,
            chat_interface_group,
            standard_interface_group,
            prompt_box,
            inf_btn,
            data_dir_inf,
            out_dir_inf,
            num_samples_box,
            max_new_tokens_box,
            temperature_box,
            top_k_box,
            dtype_box_inf,
            device_box_inf,
            seed_box_inf,
            inf_output,
            inf_advanced_output,
            inf_send_btn,
            inf_user_input,
            chatbot,
            model_dropdown,
            inf_system_prompt,
            chat_advanced_output,
            inf_clear_btn,
        )

        # ------------------------------------------------------------------
        # Call backs: model selection, reset, delete
        outputs_for_model_select_and_delete = [
            new_model_chk,
            model_name_box,
            data_dir_box,
            out_dir_box,
            plot_interval_box,
            log_interval_box,
            num_eval_seeds_box,
            save_best_val_ckpt_box,
            init_from_box,
            grad_acc_box,
            batch_size_box,
            block_size_box,
            n_layer_box,
            n_head_box,
            n_embd_box,
            dropout_box,
            bias_box,
            lr_box,
            max_iters_box,
            weight_decay_box,
            beta1_box,
            beta2_box,
            lr_scheduler_box,
            warmup_box,
            lr_decay_box,
            min_lr_box,
            step_size_box,
            step_gamma_box,
            polynomial_power_box,
            backend_box,
            device_box,
            dtype_box,
            compile_box,
            seed_box,
            save_interval_box,
            # Self-attention parameters
            use_self_attention_box,
            ffn_hidden_mult_box,
            qkv_bias_box,
            attn_dropout_box,
            resid_dropout_box,
            ln_eps_box,
            init_std_box,
            use_flash_attn_box,
            pos_encoding_type_box,
            rope_base_box,
            # New optimized parameters
            rope_cache_size_box,
            alibi_bias_scale_box,
            ffn_activation_box,
            attention_scale_factor_box,
            gradient_checkpointing_box,
            cache_strategy_box,
            max_cache_size_box,
            strict_validation_box,
            fallback_on_error_box,
            train_plot,
            train_log,
            data_dir_inf,
            out_dir_inf,
            prompt_box,
            num_samples_box,
            max_new_tokens_box,
            temperature_box,
            top_k_box,
            dtype_box_inf,
            device_box_inf,
            seed_box_inf,
            inf_btn,
            inf_output,
            inf_advanced_output,
            chatbot,
            inf_system_prompt,
            chat_advanced_output,
            # Comparison tab components
            comp_left_model,
            comp_right_model,
            comp_left_params,
            comp_right_params,
            comp_left_plot,
            comp_right_plot,
            comp_left_history,
            comp_right_history,
            comp_left_num_samples,
            comp_left_max_tokens,
            comp_left_temperature,
            comp_left_top_k,
            comp_left_dtype,
            comp_left_seed,
            comp_right_num_samples,
            comp_right_max_tokens,
            comp_right_temperature,
            comp_right_top_k,
            comp_right_dtype,
            comp_right_seed,
            comp_prompt,
            comp_generate_btn,
            comp_left_output,
            comp_right_output,
            # Hidden comparison fields
            comp_left_data_dir,
            comp_left_out_dir,
            comp_right_data_dir,
            comp_right_out_dir,
            # SFT params (per-model persistence)
            sft_init_from,
            sft_save_best_loss_ckpt,
            sft_epochs,
            sft_learning_rate,
            sft_batch_size,
            sft_max_seq_length,
            sft_gradient_accumulation,
            sft_lr_scheduler,
            sft_warmup_iters,
            sft_lr_decay_iters,
            sft_min_lr,
            sft_step_size,
            sft_step_gamma,
            sft_polynomial_power,
            sft_label_smoothing,
            sft_freeze_layers,
            sft_grad_clip,
            sft_weight_decay,
            sft_system_prompt,
            # SFT training history (log and plot)
            sft_log,
            sft_plot,
        ]

        # ------------------------------------------------------------------
        # Call backs: language switch
        lang_select_outputs = [
            lang_select,
            data_process_tab,
            train_tab,
            inf_tab,
            comp_tab,
            model_dropdown,
            refresh_models_btn,
            delete_model_btn,
            new_model_chk,
            model_name_box,
            input_text,
            txt_dir,
            no_val_set,
            use_gpt2,
            train_split,
            num_proc,
            process_btn,
            process_output,
            train_params_title_md,
            data_dir_box,
            out_dir_box,
            backend_box,
            device_box,
            dtype_box,
            compile_box,
            plot_interval_box,
            log_interval_box,
            num_eval_seeds_box,
            save_best_val_ckpt_box,
            init_from_box,
            seed_box,
            grad_acc_box,
            batch_size_box,
            block_size_box,
            n_layer_box,
            n_head_box,
            n_embd_box,
            dropout_box,
            bias_box,
            lr_box,
            max_iters_box,
            weight_decay_box,
            beta1_box,
            beta2_box,
            lr_scheduler_box,
            warmup_box,
            lr_decay_box,
            min_lr_box,
            step_size_box,
            step_gamma_box,
            polynomial_power_box,
            save_interval_box,
            train_btn,
            stop_btn,
            train_log,
            train_plot,
            data_dir_inf,
            out_dir_inf,
            prompt_box,
            num_samples_box,
            max_new_tokens_box,
            temperature_box,
            top_k_box,
            dtype_box_inf,
            device_box_inf,
            seed_box_inf,
            inf_btn,
            inf_output,
            inf_advanced_output,
            advanced_accordion,
            # Chat mode components
            inf_chat_mode,
            chatbot,
            inf_system_prompt,
            inf_user_input,
            inf_send_btn,
            inf_clear_btn,
            chat_advanced_accordion,
            chat_advanced_output,
            # Comparison tab components
            comp_left_model,
            comp_right_model,
            comp_left_params,
            comp_right_params,
            comp_left_plot,
            comp_right_plot,
            comp_left_history,
            comp_right_history,
            comp_left_num_samples,
            comp_left_max_tokens,
            comp_left_temperature,
            comp_left_top_k,
            comp_left_dtype,
            comp_left_seed,
            comp_right_num_samples,
            comp_right_max_tokens,
            comp_right_temperature,
            comp_right_top_k,
            comp_right_dtype,
            comp_right_seed,
            comp_prompt,
            comp_generate_btn,
            comp_left_output,
            comp_right_output,
            # Self-attention parameters
            self_attn_accordion,
            use_self_attention_box,
            ffn_hidden_mult_box,
            qkv_bias_box,
            attn_dropout_box,
            resid_dropout_box,
            ln_eps_box,
            init_std_box,
            use_flash_attn_box,
            pos_encoding_type_box,
            rope_base_box,
            # New optimized parameters
            rope_cache_size_box,
            alibi_bias_scale_box,
            ffn_activation_box,
            attention_scale_factor_box,
            gradient_checkpointing_box,
            cache_strategy_box,
            max_cache_size_box,
            strict_validation_box,
            fallback_on_error_box,
            # SFT tab components
            sft_tab,
            sft_title_md,
            sft_basic_params_accordion,
            sft_optim_params_accordion,
            sft_scheduler_accordion,
            sft_dataset_example,
            sft_dataset_file,
            sft_dataset_dir,
            sft_format_status,
            sft_validate_btn,
            sft_init_from,
            sft_save_best_loss_ckpt,
            sft_epochs,
            sft_learning_rate,
            sft_batch_size,
            sft_max_seq_length,
            sft_gradient_accumulation,
            sft_lr_scheduler,
            sft_warmup_iters,
            sft_lr_decay_iters,
            sft_min_lr,
            sft_step_size,
            sft_step_gamma,
            sft_polynomial_power,
            sft_label_smoothing,
            sft_freeze_layers,
            sft_grad_clip,
            sft_weight_decay,
            sft_system_prompt,
            sft_start_btn,
            sft_stop_btn,
            sft_progress,
            sft_log,
            sft_plot,
        ]
        bind_model_management(
            model_dropdown,
            delete_model_btn,
            refresh_models_btn,
            outputs_for_model_select_and_delete,
            comp_left_model,
            comp_right_model,
        )

        # ------------------------------------------------------------------
        # Call backs: language switch
        bind_language(lang_select, lang_select_outputs)

        # ------------------------------------------------------------------ #
        # Call backs: comparison page
        # ------------------------------------------------------------------ #
        bind_comparison(
            comp_left_model,
            comp_right_model,
            comp_left_params,
            comp_left_plot,
            comp_left_history,
            comp_left_data_dir,
            comp_left_out_dir,
            comp_right_params,
            comp_right_plot,
            comp_right_history,
            comp_right_data_dir,
            comp_right_out_dir,
            comp_generate_btn,
            comp_prompt,
            comp_left_num_samples,
            comp_left_max_tokens,
            comp_left_temperature,
            comp_left_top_k,
            comp_left_dtype,
            comp_left_seed,
            comp_right_num_samples,
            comp_right_max_tokens,
            comp_right_temperature,
            comp_right_top_k,
            comp_right_dtype,
            comp_right_seed,
            comp_left_output,
            comp_right_output,
        )

        # ------------------------------------------------------------------
        # SFT Callbacks
        # ------------------------------------------------------------------
        bind_sft(
            sft_validate_btn,
            sft_dataset_file,
            sft_dataset_dir,
            lang_select,
            sft_format_status,
            sft_dataset_state,
            sft_start_btn,
            model_dropdown,
            sft_epochs,
            sft_learning_rate,
            sft_batch_size,
            sft_max_seq_length,
            sft_gradient_accumulation,
            sft_lr_scheduler,
            sft_warmup_iters,
            sft_lr_decay_iters,
            sft_min_lr,
            sft_step_size,
            sft_step_gamma,
            sft_polynomial_power,
            sft_label_smoothing,
            sft_freeze_layers,
            sft_grad_clip,
            sft_weight_decay,
            sft_system_prompt,
            sft_init_from,
            sft_save_best_loss_ckpt,
            sft_progress,
            sft_log,
            sft_plot,
            sft_stop_btn,
        )

    return demo


# ----------------- Launch -------------------
if __name__ == "__main__":
    app = build_app_interface()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False)
