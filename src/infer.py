# src/infer.py

import os
import pickle
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.config import DEFAULT_CONFIG
from src.db_manager import DBManager
from src.gpt_model import GPTConfig, GPT

dbm = DBManager()

def safe_decode(decode_func, tokens, fallback_char=""):
    """
    安全解码函数，处理可能的解码错误
    """
    try:
        return decode_func(tokens)
    except Exception:
        # 如果解码失败，返回fallback字符
        return fallback_char

def generate_text(
    data_dir,
    out_dir,
    prompt=DEFAULT_CONFIG["inference"]["prompt"],
    num_samples=DEFAULT_CONFIG["inference"]["num_samples"],
    max_new_tokens=DEFAULT_CONFIG["inference"]["max_new_tokens"],
    temperature=DEFAULT_CONFIG["inference"]["temperature"],
    top_k=DEFAULT_CONFIG["inference"]["top_k"],
    seed=DEFAULT_CONFIG["inference"]["seed"],
    device=DEFAULT_CONFIG["inference"]["device"],
    dtype=DEFAULT_CONFIG["inference"]["dtype"],
    compile_model=DEFAULT_CONFIG["inference"]["compile_model"]
):
    """
    Generates text from a single checkpoint and把推理配置/历史写入数据库。
    若 out_dir 以 .pt 结尾，则直接当作 ckpt 路径；否则默认 out_dir/ckpt.pt。
    """
    # ---------------- 0. Database: ensure model_id & record config ---------------- #
    # -- DB Integration --
    ckpt_dir = out_dir if out_dir.endswith('.pt') else os.path.join(out_dir, 'ckpt.pt')
    model_dir_for_db = os.path.dirname(ckpt_dir)  # 用目录定位模型
    model_name_for_db = os.path.basename(model_dir_for_db) or "new_model"
    model_id = dbm.get_model_id_by_dir(model_dir_for_db)
    if model_id is None:
        model_id = dbm.register_model(model_name_for_db, model_dir_for_db)

    inf_cfg_dict = dict(
        data_dir=data_dir,
        out_dir=out_dir,
        prompt=prompt,
        num_samples=num_samples,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        device=device,
        dtype=dtype,
        compile_model=compile_model
    )
    dbm.save_inference_config(model_id, inf_cfg_dict)
    # ------------------------------------------------------------------ #

    if not prompt.strip():
        yield "Prompt is empty, please provide a starting text."
        return

    try:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        device_type = 'cuda' if 'cuda' in device else 'cpu'
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
        ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

        # ----------- 1. Analyze ckpt path -----------
        ckpt_path = ckpt_dir
        if not os.path.exists(ckpt_path):
            err = f"Error: checkpoint not found at {ckpt_path}."
            yield err
            return

        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)

        model.eval()
        model.to(device)
        if compile_model:
            model = torch.compile(model)

        meta_path = os.path.join(data_dir, 'meta.pkl')
        if not os.path.exists(meta_path):
            yield f"Error: meta.pkl not found at {meta_path}."
            return

        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)

        tokenizer_type = meta.get('tokenizer', 'character level')
        stoi, itos = meta['stoi'], meta['itos']

        # 如果meta中有old2new映射，说明使用了tokenizer
        if 'old2new' in meta:
            old2new = meta['old2new']
            new2old = {new: old for old, new in old2new.items()}

            # 根据tokenizer类型选择不同的编解码方法
            if tokenizer_type == 'custom_json':
                try:
                    from tokenizers import Tokenizer
                    tokenizer_path = os.path.join(os.path.dirname(os.path.dirname(data_dir)), "assets/tokenizer.json")
                    if os.path.exists(tokenizer_path):
                        tokenizer = Tokenizer.from_file(tokenizer_path)

                        def encode(s):
                            ids = tokenizer.encode(s).ids
                            # 原始ID映射到训练时使用的新ID
                            return [old2new.get(id, 0) for id in ids if id in old2new]

                        def decode(l):
                            # 新ID映射回原始ID
                            original_ids = [new2old.get(id, 0) for id in l]
                            return safe_decode(tokenizer.decode, original_ids)
                    else:
                        # 找不到tokenizer文件，使用meta中的映射
                        def encode(s):
                            return [stoi.get(ch, 0) for ch in s]
                        def decode(l):
                            return ''.join([itos.get(i, '') for i in l])
                except ImportError:
                    # 找不到tokenizers库，使用meta中的映射
                    def encode(s):
                        return [stoi.get(ch, 0) for ch in s]
                    def decode(l):
                        return ''.join([itos.get(i, '') for i in l])

            elif tokenizer_type == 'gpt2':
                import tiktoken
                enc = tiktoken.get_encoding("gpt2")

                def encode(s):
                    ids = enc.encode(s, allowed_special={"<|endoftext|>"})
                    # 原始ID映射到训练时使用的新ID
                    return [old2new.get(id, 0) for id in ids if id in old2new]

                def decode(l):
                    # 新ID映射回原始ID
                    original_ids = [new2old.get(id, 0) for id in l]
                    return safe_decode(enc.decode, original_ids)

            else:
                # 默认方式
                def encode(s):
                    return [stoi.get(ch, 0) for ch in s]
                def decode(l):
                    return ''.join([itos.get(i, '') for i in l])
        else:
            # 字符级编码（没有ID重映射）
            def encode(s):
                return [stoi.get(ch, 0) for ch in s]
            def decode(l):
                return ''.join([itos.get(i, '') for i in l])

        xids = torch.tensor(encode(prompt), dtype=torch.long, device=device)[None, ...]
        block_size = gptconf.block_size
        if xids.size(1) > block_size:
            yield f"Error: input length ({xids.size(1)}) exceeds block size ({block_size})."
            return

        # ----------- 3. Generate text & accumulate output ------------
        accumulated_output = []
        with torch.no_grad():
            with ctx:
                for s_i in range(num_samples):
                    # 每个样本开始时输出标题
                    sample_header = f"Sample {s_i+1}:\n"
                    yield sample_header

                    idx = xids.clone()
                    # 先输出提示词部分
                    current_text = prompt
                    yield current_text
                    
                    # 用于存储当前样本的完整生成序列
                    generated_tokens = []
                    last_valid_text = prompt
                    buffer_size = 5  # 缓冲区大小，用于处理多字节字符
                    
                    for token_i in range(max_new_tokens):
                        if idx.size(1) == 0:
                            yield "Can't generate an empty sequence."
                            return

                        idx_cond = idx[:, -block_size:]
                        logits, _ = model(idx_cond)
                        logits = logits[:, -1, :] / temperature
                        if top_k is not None and top_k > 0:
                            v, _ = torch.topk(logits, top_k)
                            top_value = v[:, -1].unsqueeze(-1)
                            logits[logits < top_value] = -float('Inf')
                        probs = F.softmax(logits, dim=-1)
                        idx_next = torch.multinomial(probs, num_samples=1)
                        idx = torch.cat((idx, idx_next), dim=1)

                        # 将新生成的token添加到列表中
                        new_token = idx_next[0].item()
                        generated_tokens.append(new_token)

                        # 尝试解码当前的token序列
                        # 对于使用tokenizer的情况，使用缓冲策略
                        if 'old2new' in meta and tokenizer_type in ['custom_json', 'gpt2']:
                            # 每隔几个token或到达缓冲区大小时尝试解码
                            if len(generated_tokens) % buffer_size == 0 or token_i == max_new_tokens - 1:
                                # 解码完整的生成序列（包括prompt）
                                full_tokens = idx[0].tolist()
                                current_text = decode(full_tokens)
                                
                                # 只输出新增的有效部分
                                if len(current_text) > len(last_valid_text):
                                    new_text = current_text[len(last_valid_text):]
                                    # 检查新文本是否包含乱码
                                    if "�" not in new_text:
                                        yield new_text
                                        last_valid_text = current_text
                        else:
                            # 字符级编码，每个token都对应一个字符，可以直接解码
                            current_text = decode(idx[0].tolist())
                            new_text = current_text[len(last_valid_text):]
                            yield new_text
                            last_valid_text = current_text

                    # 样本生成完毕，确保最后的内容被完整解码
                    final_text = decode(idx[0].tolist())
                    if len(final_text) > len(last_valid_text):
                        remaining_text = final_text[len(last_valid_text):]
                        if "�" not in remaining_text:
                            yield remaining_text

                    # 保存完整样本
                    full_sample = f"{sample_header}{final_text}"
                    accumulated_output.append(full_sample)

                    # 样本之间添加分隔线
                    if s_i < num_samples - 1:
                        separator = "\n" + "-" * 30 + "\n"
                        yield separator

        final_text = "\n\n".join(accumulated_output)

        # ---------------- 3. Write inference history to DB ------------------ #
        # -- DB Integration --
        dbm.save_inference_history(model_id, final_text)

    except Exception as ex:
        yield f"An unexpected error occurred: {str(ex)}"
