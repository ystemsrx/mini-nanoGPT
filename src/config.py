# config.py
import numpy as np
from multiprocessing import cpu_count

import torch
if torch.cuda.is_available():
    selected_device = "cuda:0"
    selected_backend = "nccl"
else:
    selected_device = "cpu"
    selected_backend = "gloo"


# Define the data type.
IntegerTypes = np.uint32

# Define default configuration values.
DEFAULT_CONFIG = {
    "data_process": {
        "train_split_ratio": 0.9,
        "no_validation": False,
        "use_gpt2_tokenizer": True,
        "num_proc": cpu_count() // 2
    },
    "training": {
        "out_dir": "out",
        "plot_interval": 10,
        "log_interval": 10,
        "num_eval_seeds": 0,
        "save_best_val_checkpoint": True,
        "init_from": "scratch",
        "gradient_accumulation_steps": 1,
        "batch_size": 32,
        "block_size": 256,
        "n_layer": 6,
        "n_head": 6,
        "n_embd": 384,
        "dropout": 0.1,
        "bias": True,
        "learning_rate": 1e-3,
        "max_iters": 300,
        "weight_decay": 1e-2,
        "beta1": 0.9,
        "beta2": 0.999,
        "lr_scheduler_type": "cosine",
        "warmup_iters": 10,
        "lr_decay_iters": 300,
        "min_lr": 1e-5,
        "step_size": 150,
        "step_gamma": 0.1,
        "polynomial_power": 2.0,
        "backend": selected_backend,
        "device": selected_device,
        "dtype": "float16",
        "compile_model": False,
        "seed": 1024,
        "save_interval": 50,
        # Self-attention specific parameters
        "use_self_attention": False,
        "ffn_hidden_mult": 4,
        "qkv_bias": True,
        "attn_dropout": 0.1,
        "resid_dropout": 0.1,
        "ln_eps": 1e-5,
        "init_std": 0.02,
        "use_flash_attn": False,
        "pos_encoding_type": "rope",
        "rope_base": 10000,
        "rope_cache_size": 1024,  # Dynamic RoPE cache size (None for auto)
        "alibi_bias_scale": 1.0,  # ALiBi bias scaling factor
        "ffn_activation": "gelu",  # FFN activation function: gelu, relu, swish
        "attention_scale_factor": 1.0,  # Additional attention scaling
        "gradient_checkpointing": False,  # Memory-efficient training
        # Memory management
        "cache_strategy": "adaptive",  # Cache allocation strategy: adaptive, fixed, minimal
        "max_cache_size": 4096,  # Maximum cache size for dynamic allocation
        # Error handling
        "strict_validation": True,  # Enable strict input validation
        "fallback_on_error": True  # Fallback to basic implementations on error
    },
    "inference": {
        "out_dir": "out",
        "prompt": "",
        "num_samples": 3,
        "max_new_tokens": 64,
        "temperature": 0.7,
        "top_k": 50,
        "seed": 1024,
        "device": selected_device,
        "dtype": "float16",
        "compile_model": False
    },
    "sft": {
        "dataset_path": "",
        "init_from": "scratch",
        "save_best_loss_checkpoint": True,
        "epochs": 3,
        "learning_rate": 2e-5,
        "batch_size": 4,
        "max_seq_length": 512,
        "gradient_accumulation_steps": 4,
        "lr_scheduler_type": "cosine",
        "warmup_iters": 0,
        "lr_decay_iters": 0,
        "min_lr": 1e-6,
        "step_size": 50,
        "step_gamma": 0.1,
        "polynomial_power": 2.0,
        "warmup_ratio": 0.1,
        "label_smoothing": 0.0,
        "freeze_layers": 0,
        "grad_clip": 1.0,
        "weight_decay": 0.01,
        "save_steps": 100,
        "logging_steps": 10,
        "system_prompt": "You are a helpful assistant."
    }
}

# Multilingual support
LANG_JSON = {
    "en": {
        "app_title": "Mini Nano GPT",
        "language_label": "Language",
        "data_process_tab": "Data Processing",
        "train_tab": "Pre-training",
        "infer_tab": "Inference",
        "compare_tab": "Comparison",
        "model_tab": "Model Management",

        "registered_models": "Registered Models",
        "refresh_tables": "Refresh",
        "delete_selected_model": "Delete Selected Model",

        "new_model": "New Model",
        "model_name": "Model Name",

        "dp_paste_text": "Paste Text",
        "dp_txt_dir": "TXT Directory (Optional)",
        "dp_raw_dir": "Raw Data Directory",
        "dp_processed_dir": "Processed Data Directory",
        "dp_train_split": "Training Split Ratio",
        "dp_no_val_set": "Do not use validation set",
        "dp_use_gpt2_tokenizer": "Use GPT-2/Qwen Tokenizer",
        "dp_num_proc": "Number of Processes",
        "dp_start_btn": "Start Processing",
        "dp_result": "Processing Result",

        "train_params_title": "Training Parameters",
        "train_data_dir": "Data Directory (where train.bin/val.bin)",
        "train_out_dir": "Output Directory",
        "train_eval_interval": "Plot Interval",
        "train_log_interval": "Logging Interval",
        "train_num_eval_seeds": "Number of Evaluation Seeds",
        "train_save_best_val_ckpt": "Save Best Val Loss Checkpoint",
        "train_init_from": "Initialization Source",
        "train_gas": "Gradient Accumulation Steps",
        "train_batch_size": "Batch Size",
        "train_block_size": "Block Size",
        "train_n_layer": "Number of Layers",
        "train_n_head": "Number of Attention Heads",
        "train_n_embd": "Embedding Dimension",
        "train_dropout": "Dropout Rate",
        "train_bias": "Use Bias",
        "train_lr": "Learning Rate",
        "train_max_iters": "Maximum Iterations",
        "train_weight_decay": "Weight Decay",
        "train_beta1": "Beta 1",
        "train_beta2": "Beta 2",
        "train_lr_scheduler": "Learning Rate Scheduler",
        "train_warmup_iters": "Warmup Iterations",
        "train_lr_decay_iters": "Learning Rate Decay Iterations",
        "train_min_lr": "Minimum Learning Rate",
        "train_step_size": "Step Size",
        "train_step_gamma": "Step Gamma",
        "train_poly_power": "Polynomial Power",
        "train_backend": "Backend",
        "train_device": "Device",
        "train_dtype": "Data Type",
        "train_compile_model": "Compile Model",
        "train_start_btn": "Start Training",
        "train_log": "Training Log",
        "train_plot": "Loss Curve",
        "train_seed": "Seed",
        "train_save_interval": "Save Interval (Steps)",

        # Self-attention parameters
        "train_self_attn_title": "Self-Attention Parameters",
        "train_use_self_attention": "Enable Self-Attention",
        "train_ffn_hidden_mult": "FFN Hidden Multiplier",
        "train_qkv_bias": "QKV Bias",
        "train_attn_dropout": "Attention Dropout",
        "train_resid_dropout": "Residual Dropout",
        "train_ln_eps": "Layer Norm Epsilon",
        "train_init_std": "Weight Init Std",
        "train_use_flash_attn": "Use Flash Attention",
        "train_pos_encoding_type": "Position Encoding",
        "train_rope_base": "RoPE Base",

        # New optimized parameters
        "train_rope_cache_size": "RoPE Cache Size",
        "train_alibi_bias_scale": "ALiBi Bias Scale",
        "train_ffn_activation": "FFN Activation",
        "train_attention_scale_factor": "Attention Scale",
        "train_gradient_checkpointing": "Gradient Checkpointing",
        "train_cache_strategy": "Cache Strategy",
        "train_max_cache_size": "Max Cache Size",
        "train_strict_validation": "Strict Validation",
        "train_fallback_on_error": "Fallback on Error",

        "inf_out_dir": "Model Directory (ckpt.pt)",
        "inf_prompt": "Prompt",
        "inf_num_samples": "Number of Samples",
        "inf_max_new_tokens": "Maximum New Tokens",
        "inf_temperature": "Temperature",
        "inf_top_k": "Top K",
        "inf_dtype": "Data Type",
        "inf_start_btn": "Generate",
        "inf_stop_btn": "Stop",
        "inf_result": "Generation Result",
        "inf_seed": "Seed",
        "inf_device": "Inference Device",
        "inf_advanced_output": "Detailed Information",
        "inf_token_position": "Position",
        "inf_selected_token": "Selected",
        "inf_top_candidates": "Top 5 Candidates",
        "inf_probability": "Probability",
        "inf_chat_tokenization": "User Input Tokenization",
        "inf_chat_advanced": "Chat Detailed Information",
        "inf_token_text": "Token",
        "inf_token_id": "Token ID",
        "inf_in_vocab": "In Vocab",
        "stop_btn": "Stop Training",

        "model_new": "Create New Model",
        "model_name": "Model Name",
        "model_description": "Description",
        "model_select": "Select Model",
        "model_create_btn": "Create",
        "model_delete_btn": "Delete",
        "model_save_btn": "Save",
        "model_list": "Model List",
        "model_current": "Current Model",
        "model_id": "ID",
        "model_create_time": "Created",
        "model_update_time": "Updated",
        "model_dir": "Directory",
        
        "compare_left_model": "Left",
        "compare_right_model": "Right",
        "compare_model_params": "Model Parameters",
        "compare_loss_curve": "Loss Curve",
        "compare_inference_history": "Inference History",
        "compare_inference_playground": "Playground",
        "compare_inference_params": "Inference Parameters",
        "compare_chat_mode": "Chat Mode (Both SFT)",
        "compare_generate_btn": "Generate",
        "compare_shared_prompt": "Shared Prompt",
        "compare_left_output": "Left Model Output",
        "compare_right_output": "Right Model Output",

        # SFT Tab
        "sft_tab": "SFT",
        "sft_title": "Supervised Fine-Tuning (SFT)",
        "sft_dataset_example": "Example Dataset Format",
        "sft_dataset_file": "Dataset File (JSON)",
        "sft_dataset_dir": "Dataset Directory",
        "sft_format_status": "Format Validation",
        "sft_validate_btn": "🔍 Validate Dataset",
        "sft_basic_params": "SFT Basic Parameters",
        "sft_optim_params": "Optimization & Regularization",
        "sft_scheduler_params": "Learning Rate Scheduler",
        "sft_epochs": "Epochs",
        "sft_learning_rate": "Learning Rate",
        "sft_batch_size": "Batch Size",
        "sft_max_seq_length": "Max Sequence Length",
        "sft_gradient_accumulation": "Gradient Accumulation Steps",
        "sft_init_from": "Initialization Source",
        "sft_save_best_loss_ckpt": "Save Best Loss Checkpoint",
        "sft_lr_scheduler": "Learning Rate Scheduler",
        "sft_warmup_iters": "Warmup Steps",
        "sft_lr_decay_iters": "LR Decay Steps",
        "sft_min_lr": "Minimum Learning Rate",
        "sft_step_size": "Step Size",
        "sft_step_gamma": "Step Gamma",
        "sft_poly_power": "Polynomial Power",
        "sft_label_smoothing": "Label Smoothing",
        "sft_freeze_layers": "Freeze Layers",
        "sft_grad_clip": "Gradient Clipping",
        "sft_weight_decay": "Weight Decay",
        "sft_system_prompt": "System Prompt",
        "sft_start_btn": "Start SFT Training",
        "sft_stop_btn": "Stop SFT",
        "sft_progress": "SFT Progress",
        "sft_log": "SFT Training Log",
        "sft_plot": "SFT Loss Curve",
        "sft_result": "SFT Result",
        "sft_valid_format": "✅ Valid Alpaca Format",
        "sft_invalid_format": "❌ Invalid Format",
        "sft_no_dataset": "No dataset loaded",
        "sft_dataset_example_json": (
            "[\n"
            "  {\n"
            "    \"instruction\": \"Why is the sky blue?\",\n"
            "    \"input\": \"\",\n"
            "    \"output\": \"The sky appears blue mainly due to Rayleigh scattering in the atmosphere...\"\n"
            "  },\n"
            "  {\n"
            "    \"instruction\": \"Please explain this concept:\",\n"
            "    \"input\": \"Relativity\",\n"
            "    \"output\": \"Relativity is a physical theory proposed by Albert Einstein...\"\n"
            "  },\n"
            "  {\n"
            "    \"instruction\": \"Summarize the following text:\",\n"
            "    \"input\": \"Neural networks are inspired by the brain and consist of layers of interconnected neurons.\",\n"
            "    \"output\": \"Neural networks are brain-inspired layered models made of connected neurons.\"\n"
            "  },\n"
            "  {\n"
            "    \"instruction\": \"Write a short greeting:\",\n"
            "    \"input\": \"\",\n"
            "    \"output\": \"Hello! How can I help you today?\"\n"
            "  }\n"
            "]"
        ),

        # Chat Mode
        "inf_chat_mode": "Chat Mode (for SFT models)",
        "inf_chat_history": "Conversation History",
        "inf_user_input": "Your Message",
        "inf_send_btn": "Send",
        "inf_clear_chat": "Clear Chat",
        "inf_system_prompt": "System Prompt"
    },
    "zh": {
        "app_title": "Mini Nano GPT",
        "language_label": "语言",
        "data_process_tab": "数据处理",
        "train_tab": "预训练",
        "infer_tab": "推理",
        "compare_tab": "对比",
        "model_tab": "模型管理",

        "registered_models": "已注册模型",
        "refresh_tables": "刷新",
        "delete_selected_model": "删除选中模型",

        "new_model": "新模型",
        "model_name": "模型名称",

        "dp_paste_text": "粘贴文本",
        "dp_txt_dir": "TXT文件目录（可选）",
        "dp_raw_dir": "原始数据目录",
        "dp_processed_dir": "处理后数据目录",
        "dp_train_split": "训练集比例 (Training Split Ratio)",
        "dp_no_val_set": "暂不需要验证集",
        "dp_use_gpt2_tokenizer": "使用 GPT-2/Qwen 分词器 (Use GPT-2/Qwen Tokenizer)",
        "dp_num_proc": "进程数 (Number of processes)",
        "dp_start_btn": "开始处理",
        "dp_result": "处理结果",

        "train_params_title": "训练参数 (Training Parameters)",
        "train_data_dir": "数据目录（包含 train.bin/val.bin）",
        "train_out_dir": "输出目录",
        "train_eval_interval": "评估间隔 (Plot Interval)",
        "train_log_interval": "日志间隔 (Logging Interval)",
        "train_num_eval_seeds": "评估种子数量 (Number of Evaluation Seeds)",
        "train_save_best_val_ckpt": "保存验证损失最佳点 (Save Best Val Loss Checkpoint)",
        "train_init_from": "初始化方式 (Initialization Source)",
        "train_gas": "梯度累积步数 (Gradient Accumulation Steps)",
        "train_batch_size": "批量大小 (Batch Size)",
        "train_block_size": "块/上下文大小 (Block Size)",
        "train_n_layer": "层数 (Number of Layers)",
        "train_n_head": "头数 (Number of Attention Heads)",
        "train_n_embd": "嵌入维度 (Embedding Dimension)",
        "train_dropout": "丢弃率 (Dropout Rate)",
        "train_bias": "是否使用偏置？ (Use Bias)",
        "train_lr": "学习率 (Learning Rate)",
        "train_max_iters": "最大迭代次数 (Maximum Iterations)",
        "train_weight_decay": "权重衰减 (Weight Decay)",
        "train_beta1": "β1 (Beta 1)",
        "train_beta2": "β2 (Beta 2)",
        "train_lr_scheduler": "学习率调度器 (Learning Rate Scheduler)",
        "train_warmup_iters": "预热迭代次数 (Warmup Iterations)",
        "train_lr_decay_iters": "学习率衰减迭代次数 (Learning Rate Decay Iterations)",
        "train_min_lr": "最小学习率 (Minimum Learning Rate)",
        "train_step_size": "步长 (Step Size)",
        "train_step_gamma": "衰减率 (Step Gamma)",
        "train_poly_power": "衰减指数 (Polynomial Power)",
        "train_backend": "后端 (Backend)",
        "train_device": "设备 (Device)",
        "train_dtype": "数据类型 (Data Type)",
        "train_compile_model": "编译模型 (Compile Model)",
        "train_start_btn": "开始训练",
        "train_log": "训练日志 (Training Log)",
        "train_plot": "损失曲线 (Loss Curve)",
        "train_seed": "种子 (Seed)",
        "train_save_interval": "保存间隔 (Save Interval)",

        # Self-attention parameters
        "train_self_attn_title": "自注意力参数 (Self-Attention Parameters)",
        "train_use_self_attention": "启用自注意力 (Enable Self-Attention)",
        "train_ffn_hidden_mult": "FFN隐藏层倍数 (FFN Hidden Multiplier)",
        "train_qkv_bias": "QKV偏置 (QKV Bias)",
        "train_attn_dropout": "注意力丢弃率 (Attention Dropout)",
        "train_resid_dropout": "残差丢弃率 (Residual Dropout)",
        "train_ln_eps": "层归一化精度 (Layer Norm Epsilon)",
        "train_init_std": "权重初始化标准差 (Weight Init Std)",
        "train_use_flash_attn": "使用 Flash Attention (Use Flash Attention)",
        "train_pos_encoding_type": "位置编码类型 (Position Encoding)",
        "train_rope_base": "RoPE基数 (RoPE Base)",

        # New optimized parameters
        "train_rope_cache_size": "RoPE缓存大小 (RoPE Cache Size)",
        "train_alibi_bias_scale": "ALiBi偏置缩放 (ALiBi Bias Scale)",
        "train_ffn_activation": "FFN激活函数 (FFN Activation)",
        "train_attention_scale_factor": "注意力缩放因子 (Attention Scale)",
        "train_gradient_checkpointing": "梯度检查点 (Gradient Checkpointing)",
        "train_cache_strategy": "缓存策略 (Cache Strategy)",
        "train_max_cache_size": "最大缓存大小 (Max Cache Size)",
        "train_strict_validation": "严格验证 (Strict Validation)",
        "train_fallback_on_error": "错误回退 (Fallback on Error)",

        "inf_out_dir": "模型目录（ckpt.pt）",
        "inf_prompt": "提示词 (Prompt)",
        "inf_num_samples": "生成样本数 (Number of Samples)",
        "inf_max_new_tokens": "最多生成标记数 (Maximum New Tokens)",
        "inf_temperature": "温度 (Temperature)",
        "inf_top_k": "TOP K",
        "inf_dtype": "数据类型 (Data Type)",
        "inf_start_btn": "开始生成",
        "inf_stop_btn": "终止",
        "inf_result": "生成结果",
        "inf_seed": "种子 (Seed)",
        "inf_device": "推理设备 (Inference Device)",
        "inf_advanced_output": "详细信息",
        "inf_token_position": "位置",
        "inf_selected_token": "已选token",
        "inf_top_candidates": "前5候选",
        "inf_probability": "概率",
        "inf_chat_tokenization": "用户输入分词",
        "inf_chat_advanced": "对话详细信息",
        "inf_token_text": "Token",
        "inf_token_id": "Token ID",
        "inf_in_vocab": "词表内",
        "stop_btn": "停止训练",

        "model_new": "创建新模型",
        "model_name": "模型名称",
        "model_description": "描述",
        "model_select": "选择模型",
        "model_create_btn": "创建",
        "model_delete_btn": "删除",
        "model_save_btn": "保存",
        "model_list": "模型列表",
        "model_current": "当前模型",
        "model_id": "ID",
        "model_create_time": "创建时间",
        "model_update_time": "更新时间",
        "model_dir": "目录",
        
        "compare_left_model": "左侧",
        "compare_right_model": "右侧",
        "compare_model_params": "模型参数 (Model Parameters)",
        "compare_loss_curve": "损失曲线 (Loss Curve)",
        "compare_inference_history": "推理历史",
        "compare_inference_playground": "Playground",
        "compare_inference_params": "推理参数",
        "compare_chat_mode": "对话模式（需两个模型都SFT）",
        "compare_generate_btn": "生成 (Generate)",
        "compare_shared_prompt": "提示词 (Shared Prompt)",
        "compare_left_output": "左侧模型输出",
        "compare_right_output": "右侧模型输出",

        # SFT Tab
        "sft_tab": "SFT微调",
        "sft_title": "监督微调 (SFT)",
        "sft_dataset_example": "示例数据集格式",
        "sft_dataset_file": "数据集文件 (JSON)",
        "sft_dataset_dir": "数据集目录",
        "sft_format_status": "格式验证",
        "sft_validate_btn": "🔍 验证数据集",
        "sft_basic_params": "SFT基础参数",
        "sft_optim_params": "优化与正则化",
        "sft_scheduler_params": "学习率调度器",
        "sft_epochs": "训练轮数 (Epochs)",
        "sft_learning_rate": "学习率 (Learning Rate)",
        "sft_batch_size": "批量大小 (Batch Size)",
        "sft_max_seq_length": "最大序列长度 (Max Sequence Length)",
        "sft_gradient_accumulation": "梯度累积步数 (Gradient Accumulation)",
        "sft_init_from": "初始化方式 (Initialization Source)",
        "sft_save_best_loss_ckpt": "保存损失最佳点 (Save Best Loss Checkpoint)",
        "sft_lr_scheduler": "学习率调度器",
        "sft_warmup_iters": "预热步数 (Warmup Steps)",
        "sft_lr_decay_iters": "学习率衰减步数 (LR Decay Steps)",
        "sft_min_lr": "最小学习率 (Minimum LR)",
        "sft_step_size": "阶梯步长 (Step Size)",
        "sft_step_gamma": "阶梯衰减 (Step Gamma)",
        "sft_poly_power": "多项式幂 (Polynomial Power)",
        "sft_label_smoothing": "标签平滑 (Label Smoothing)",
        "sft_freeze_layers": "层冻结 (Freeze Layers)",
        "sft_grad_clip": "梯度裁剪 (Gradient Clipping)",
        "sft_weight_decay": "权重衰减 (Weight Decay)",
        "sft_system_prompt": "系统提示词 (System Prompt)",
        "sft_start_btn": "开始SFT训练",
        "sft_stop_btn": "停止SFT",
        "sft_progress": "SFT进度",
        "sft_log": "SFT训练日志",
        "sft_plot": "SFT损失曲线",
        "sft_result": "SFT结果",
        "sft_valid_format": "✅ Alpaca格式有效",
        "sft_invalid_format": "❌ 格式无效",
        "sft_no_dataset": "未加载数据集",
        "sft_dataset_example_json": (
            "[\n"
            "  {\n"
            "    \"instruction\": \"天空为什么是蓝色的？\",\n"
            "    \"input\": \"\",\n"
            "    \"output\": \"天空之所以呈现蓝色，主要是由于大气散射现象...\"\n"
            "  },\n"
            "  {\n"
            "    \"instruction\": \"请解释此概念：\",\n"
            "    \"input\": \"相对论\",\n"
            "    \"output\": \"相对论是由阿尔伯特·爱因斯坦提出的物理学理论...\"\n"
            "  },\n"
            "  {\n"
            "    \"instruction\": \"请总结下面这段话：\",\n"
            "    \"input\": \"神经网络受大脑启发，由多层相互连接的神经元组成。\",\n"
            "    \"output\": \"神经网络是由多层互联神经元构成的脑启发模型。\"\n"
            "  },\n"
            "  {\n"
            "    \"instruction\": \"写一句简短的问候：\",\n"
            "    \"input\": \"\",\n"
            "    \"output\": \"你好！很高兴为你提供帮助。\"\n"
            "  }\n"
            "]"
        ),

        # Chat Mode
        "inf_chat_mode": "对话模式（用于SFT模型）",
        "inf_chat_history": "对话历史",
        "inf_user_input": "您的消息",
        "inf_send_btn": "发送",
        "inf_clear_chat": "清空对话",
        "inf_system_prompt": "系统提示词"
    }
}
