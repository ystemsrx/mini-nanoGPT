# src/gpt_self_attn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional

class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    def __init__(self, dim, max_seq_len, base=10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Pre-compute theta values
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute position encodings
        seq_len = max_seq_len
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        # Create cos and sin with proper dimensions
        self.register_buffer('cos_cached', freqs.cos())
        self.register_buffer('sin_cached', freqs.sin())
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        cos = self.cos_cached[:seq_len]  # [seq_len, dim//2]
        sin = self.sin_cached[:seq_len]  # [seq_len, dim//2]
        return cos, sin

def apply_rope(x, cos, sin):
    """Apply RoPE to input tensor"""
    # x: [batch, n_head, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim//2]
    
    # Split x into even and odd components
    x1 = x[..., 0::2]  # [batch, n_head, seq_len, head_dim//2]
    x2 = x[..., 1::2]  # [batch, n_head, seq_len, head_dim//2]
    
    # Reshape cos and sin to broadcast correctly
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    
    # Apply rotation
    x1_new = x1 * cos - x2 * sin
    x2_new = x1 * sin + x2 * cos
    
    # Interleave back
    return torch.cat([x1_new, x2_new], dim=-1)

class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi)"""
    def __init__(self, num_heads, max_seq_len):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Compute slopes for each head
        def get_slopes(n):
            def get_slopes_power_of_2(n):
                start = (2**(-2**-(math.log2(n)-3)))
                ratio = start
                return [start*ratio**i for i in range(n)]
            
            if math.log2(n).is_integer():
                return get_slopes_power_of_2(n)
            else:
                closest_power_of_2 = 2**math.floor(math.log2(n))
                return get_slopes_power_of_2(closest_power_of_2) + \
                       get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
        
        slopes = torch.tensor(get_slopes(num_heads)).float()
        self.register_buffer('slopes', slopes)
        
        # Pre-compute bias matrix
        arange_tensor = torch.arange(max_seq_len)
        bias = -torch.abs(arange_tensor[None, :] - arange_tensor[:, None])
        bias = bias.unsqueeze(0).repeat(num_heads, 1, 1)
        bias = bias * slopes.view(-1, 1, 1)
        self.register_buffer('bias', bias)
    
    def forward(self, seq_len):
        return self.bias[:, :seq_len, :seq_len]

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional RoPE/ALiBi"""
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Linear projections
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.qkv_bias)
        self.out_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.attn_dropout)
        self.resid_dropout = nn.Dropout(config.resid_dropout)
        
        # Position encoding
        self.pos_encoding_type = config.pos_encoding_type
        if self.pos_encoding_type == "rope":
            self.rope = RoPE(self.head_dim, config.block_size, config.rope_base)
        elif self.pos_encoding_type == "alibi":
            self.alibi = ALiBi(config.n_head, config.block_size)
        
        # Flash attention
        self.use_flash_attn = config.use_flash_attn
        
        # Causal mask
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.block_size, config.block_size))
            .view(1, 1, config.block_size, config.block_size)
        )
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply position encoding
        if self.pos_encoding_type == "rope":
            cos, sin = self.rope(q, T)
            q = apply_rope(q, cos, sin)
            k = apply_rope(k, cos, sin)
        
        # Use Flash Attention if available and enabled
        if self.use_flash_attn and hasattr(F, 'scaled_dot_product_attention'):
            try:
                # Flash attention with causal mask
                y = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=None,
                    dropout_p=self.attn_dropout.p if self.training else 0.0,
                    is_causal=True
                )
            except:
                # Fallback to manual implementation
                y = self._manual_attention(q, k, v, T)
        else:
            y = self._manual_attention(q, k, v, T)
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y
    
    def _manual_attention(self, q, k, v, T):
        # Compute attention scores
        att = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply ALiBi bias if using ALiBi
        if self.pos_encoding_type == "alibi":
            alibi_bias = self.alibi(T)
            att = att + alibi_bias.unsqueeze(0)
        
        # Apply causal mask
        att = att.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        
        # Softmax and dropout
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Apply to values
        y = att @ v
        return y

class FeedForward(nn.Module):
    """Feed-Forward Network with GELU activation"""
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_mult * config.n_embd)
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_dropout)
    
    def forward(self, x):
        return self.dropout(self.w2(F.gelu(self.w1(x))))

class TransformerBlock(nn.Module):
    """Transformer block with attention and feed-forward"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.ln_eps)
        self.attn = MultiHeadAttention(config)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.ln_eps)
        self.ffn = FeedForward(config)
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPTSelfAttnConfig:
    def __init__(
        self, 
        vocab_size, 
        block_size, 
        n_layer, 
        n_head, 
        n_embd, 
        dropout,
        bias,
        # Self-attention specific parameters
        ffn_hidden_mult,
        qkv_bias,
        attn_dropout,
        resid_dropout,
        ln_eps,
        init_std,
        use_flash_attn,
        pos_encoding_type,
        rope_base  # RoPE base parameter
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        
        # Self-attention specific
        self.ffn_hidden_mult = ffn_hidden_mult
        self.qkv_bias = qkv_bias
        self.attn_dropout = attn_dropout
        self.resid_dropout = resid_dropout
        self.ln_eps = ln_eps
        self.init_std = init_std
        self.use_flash_attn = use_flash_attn
        self.pos_encoding_type = pos_encoding_type
        self.rope_base = rope_base

class GPTSelfAttn(nn.Module):
    """GPT model with full self-attention mechanism"""
    def __init__(self, config: GPTSelfAttnConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        
        # Only use learned position embeddings if not using RoPE/ALiBi
        if config.pos_encoding_type not in ["rope", "alibi"]:
            self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.position_embedding = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.ln_eps)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Apply special scaled init to the residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('out_proj.weight') or pn.endswith('w2.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=config.init_std/math.sqrt(2 * config.n_layer))
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.block_size, f"Cannot forward sequence of length {T}, block size is only {self.block_size}"
        
        # Token embeddings
        tok_emb = self.token_embedding(idx)
        
        # Position embeddings (if not using RoPE/ALiBi)
        if self.position_embedding is not None:
            pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
            pos_emb = self.position_embedding(pos)
            x = tok_emb + pos_emb
        else:
            x = tok_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text tokens autoregressively"""
        for _ in range(max_new_tokens):
            if idx.size(1) == 0:
                raise ValueError("Input sequence is empty. Please provide at least one token.")
            
            # Crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Pluck the logits at the final step and scale by temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Apply softmax to convert to probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    
    def crop_block_size(self, block_size):
        """Adjust the model's internal block_size if needed"""
        self.config.block_size = block_size
        self.block_size = block_size

def configure_optimizers_self_attn(model, weight_decay, learning_rate, betas, device_type):
    """
    Creates and returns an AdamW optimizer for the self-attention model's parameters,
    following GPT-2 style parameter grouping with weight decay.
    """
    # Separate parameters into those that will and won't be decayed
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (torch.nn.Linear, )
    blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
    
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            
            if pn.endswith('bias'):
                # All biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # Weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # Weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
    
    # Validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )
    
    # Create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
    return optimizer 