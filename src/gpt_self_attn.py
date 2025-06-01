# src/gpt_self_attn.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Optional

class RoPE(nn.Module):
    """Rotary Position Embedding (RoPE) - now with dynamic memory allocation"""
    def __init__(self, dim, max_seq_len, base, cache_size=None):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        # Only cache a reasonable amount instead of the full max_seq_len
        self.cache_size = min(cache_size or 1024, max_seq_len)
        
        # Pre-compute theta values - these are static
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Pre-compute only a reasonable cache size instead of full max_seq_len
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
    
    def _extend_cache(self, seq_len):
        """Extend the cache if needed - dynamic allocation"""
        if seq_len <= self._cached_seq_len:
            return
        
        # Extend cache to next power of 2 or requested length, whichever is smaller
        new_cache_len = min(
            max(seq_len, self._cached_seq_len * 2),
            self.max_seq_len
        )
        
        t = torch.arange(new_cache_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        
        self._cached_cos = freqs.cos()
        self._cached_sin = freqs.sin()
        self._cached_seq_len = new_cache_len
    
    def forward(self, x, seq_len=None):
        if seq_len is None:
            seq_len = x.shape[-2]
        
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_seq_len}. "
                f"Consider increasing max_seq_len in model config."
            )
        
        # Extend cache if needed
        self._extend_cache(seq_len)
        
        cos = self._cached_cos[:seq_len]  # [seq_len, dim//2]
        sin = self._cached_sin[:seq_len]  # [seq_len, dim//2]
        return cos, sin

def apply_rope(x, cos, sin):
    """Apply RoPE to input tensor - optimized version"""
    # x: [batch, n_head, seq_len, head_dim]
    # cos, sin: [seq_len, head_dim//2]
    
    if x.size(-1) % 2 != 0:
        raise ValueError(
            f"Head dimension {x.size(-1)} must be even for RoPE. "
            f"Check your model configuration."
        )
    
    # Split x into even and odd components
    x1 = x[..., 0::2]  # [batch, n_head, seq_len, head_dim//2]
    x2 = x[..., 1::2]  # [batch, n_head, seq_len, head_dim//2]
    
    # Reshape cos and sin to broadcast correctly
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
    
    # Apply rotation
    x1_new = x1 * cos - x2 * sin
    x2_new = x1 * sin + x2 * cos
    
    # Interleave back - more efficient concatenation
    output = torch.empty_like(x)
    output[..., 0::2] = x1_new
    output[..., 1::2] = x2_new
    
    return output

class ALiBi(nn.Module):
    """Attention with Linear Biases (ALiBi) - now with dynamic allocation"""
    def __init__(self, num_heads, max_seq_len, alibi_bias_scale=1.0):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.alibi_bias_scale = alibi_bias_scale
        
        # Compute slopes for each head
        slopes = self._get_slopes(num_heads)
        self.register_buffer('slopes', slopes)
        
        # Dynamic cache for bias matrix
        self._cached_bias = None
        self._cached_seq_len = 0
    
    def _get_slopes(self, n):
        """Calculate ALiBi slopes - extracted from hardcoded implementation"""
        def get_slopes_power_of_2(n):
            # This magic number controls the slope distribution
            slope_ratio = 2**(-2**-(math.log2(n)-3))
            return [slope_ratio * (slope_ratio**i) for i in range(n)]
        
        if math.log2(n).is_integer():
            return torch.tensor(get_slopes_power_of_2(n)).float()
        else:
            # Handle non-power-of-2 head counts
            closest_power_of_2 = 2**math.floor(math.log2(n))
            base_slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = self._get_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]
            return torch.tensor(base_slopes + extra_slopes.tolist()).float()
    
    def _extend_bias_cache(self, seq_len):
        """Extend bias cache dynamically"""
        if seq_len <= self._cached_seq_len:
            return
        
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum supported length {self.max_seq_len}. "
                f"Consider increasing max_seq_len in model config."
            )
        
        # Extend to next reasonable size
        new_cache_len = min(
            max(seq_len, self._cached_seq_len * 2),
            self.max_seq_len
        )
        
        # Pre-compute bias matrix for the new cache size
        arange_tensor = torch.arange(new_cache_len, device=self.slopes.device)
        bias = -torch.abs(arange_tensor[None, :] - arange_tensor[:, None])
        bias = bias.unsqueeze(0).repeat(self.num_heads, 1, 1)
        bias = bias * (self.slopes.view(-1, 1, 1) * self.alibi_bias_scale)
        
        self._cached_bias = bias
        self._cached_seq_len = new_cache_len
    
    def forward(self, seq_len):
        """Get bias matrix for given sequence length"""
        self._extend_bias_cache(seq_len)
        return self._cached_bias[:, :seq_len, :seq_len]

class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention with optional RoPE/ALiBi - optimized version"""
    def __init__(self, config):
        super().__init__()
        
        # Basic validation
        if config.n_embd % config.n_head != 0:
            raise ValueError(
                f"Embedding dimension {config.n_embd} must be divisible by number of heads {config.n_head}. "
                f"Got remainder: {config.n_embd % config.n_head}"
            )
        
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
        
        # Position encoding with dynamic allocation
        self.pos_encoding_type = config.pos_encoding_type
        if self.pos_encoding_type == "rope":
            # Add cache_size parameter for RoPE
            cache_size = getattr(config, 'rope_cache_size', None)
            self.rope = RoPE(
                self.head_dim, 
                config.block_size, 
                config.rope_base,
                cache_size=cache_size
            )
        elif self.pos_encoding_type == "alibi":
            # Add configurable bias scale
            alibi_scale = getattr(config, 'alibi_bias_scale', 1.0)
            self.alibi = ALiBi(config.n_head, config.block_size, alibi_scale)
        
        # Flash attention
        self.use_flash_attn = config.use_flash_attn
        
        # Causal mask - only allocate when needed
        self._causal_mask = None
        self._mask_size = 0
    
    def _get_causal_mask(self, seq_len):
        """Get causal mask with dynamic allocation"""
        if seq_len <= self._mask_size and self._causal_mask is not None:
            return self._causal_mask[:, :, :seq_len, :seq_len]
        
        # Extend mask cache
        new_size = max(seq_len, self._mask_size * 2) if self._mask_size > 0 else seq_len
        device = next(self.parameters()).device
        
        self._causal_mask = torch.tril(torch.ones(new_size, new_size, device=device)).view(1, 1, new_size, new_size)
        self._mask_size = new_size
        
        return self._causal_mask[:, :, :seq_len, :seq_len]
    
    def forward(self, x):
        B, T, C = x.size()
        
        # Compute Q, K, V
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply position encoding
        if self.pos_encoding_type == "rope":
            try:
                cos, sin = self.rope(q, T)
                q = apply_rope(q, cos, sin)
                k = apply_rope(k, cos, sin)
            except Exception as e:
                raise RuntimeError(f"RoPE encoding failed: {str(e)}")
        
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
            except Exception as e:
                # Fallback to manual implementation with helpful error message
                print(f"Flash attention failed, falling back to manual implementation: {str(e)}")
                y = self._manual_attention(q, k, v, T)
        else:
            y = self._manual_attention(q, k, v, T)
        
        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.out_proj(y))
        return y
    
    def _manual_attention(self, q, k, v, T):
        """Manual attention computation with better error handling"""
        try:
            # Compute attention scores
            att = (q @ k.transpose(-2, -1)) * self.scale
            
            # Apply ALiBi bias if using ALiBi
            if self.pos_encoding_type == "alibi":
                alibi_bias = self.alibi(T)
                att = att + alibi_bias.unsqueeze(0)
            
            # Apply causal mask
            causal_mask = self._get_causal_mask(T)
            att = att.masked_fill(causal_mask == 0, float('-inf'))
            
            # Softmax and dropout
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            
            # Apply to values
            y = att @ v
            return y
        except Exception as e:
            raise RuntimeError(f"Manual attention computation failed: {str(e)}")

class FeedForward(nn.Module):
    """Feed-Forward Network with configurable activation"""
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.ffn_hidden_mult * config.n_embd)
        
        if hidden_dim <= 0:
            raise ValueError(
                f"FFN hidden dimension must be positive, got {hidden_dim}. "
                f"Check ffn_hidden_mult ({config.ffn_hidden_mult}) configuration."
            )
        
        self.w1 = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.w2 = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.resid_dropout)
        
        # Configurable activation function
        self.activation = getattr(config, 'ffn_activation', 'gelu')
        
    def forward(self, x):
        # Apply configurable activation
        if self.activation == 'gelu':
            x = F.gelu(self.w1(x))
        elif self.activation == 'relu':
            x = F.relu(self.w1(x))
        elif self.activation == 'swish':
            x = F.silu(self.w1(x))  # SiLU is the same as Swish
        else:
            # Default to GELU
            x = F.gelu(self.w1(x))
        
        return self.dropout(self.w2(x))

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
        try:
            x = x + self.attn(self.ln1(x))
            x = x + self.ffn(self.ln2(x))
            return x
        except Exception as e:
            raise RuntimeError(f"TransformerBlock forward pass failed: {str(e)}")

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
        rope_base,  # RoPE base parameter
        # New configurable parameters
        rope_cache_size=None,
        alibi_bias_scale=1.0,
        ffn_activation='gelu',
        attention_scale_factor=1.0,
        gradient_checkpointing=False
    ):
        # Basic validation
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if n_layer <= 0:
            raise ValueError(f"n_layer must be positive, got {n_layer}")
        if n_head <= 0:
            raise ValueError(f"n_head must be positive, got {n_head}")
        if n_embd <= 0:
            raise ValueError(f"n_embd must be positive, got {n_embd}")
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd ({n_embd}) must be divisible by n_head ({n_head})")
        
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
        
        # New configurable parameters
        self.rope_cache_size = rope_cache_size
        self.alibi_bias_scale = alibi_bias_scale
        self.ffn_activation = ffn_activation
        self.attention_scale_factor = attention_scale_factor
        self.gradient_checkpointing = gradient_checkpointing

class GPTSelfAttn(nn.Module):
    """GPT model with full self-attention mechanism - optimized version"""
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
        
        if T > self.block_size:
            raise ValueError(
                f"Input sequence length {T} exceeds block size {self.block_size}. "
                f"Consider using a smaller sequence or increasing block_size."
            )
        
        try:
            # Token embeddings
            tok_emb = self.token_embedding(idx)
            
            # Position embeddings (if not using RoPE/ALiBi)
            if self.position_embedding is not None:
                pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
                pos_emb = self.position_embedding(pos)
                x = tok_emb + pos_emb
            else:
                x = tok_emb
            
            # Apply transformer blocks with optional gradient checkpointing
            for i, block in enumerate(self.blocks):
                if self.config.gradient_checkpointing and self.training:
                    x = torch.utils.checkpoint.checkpoint(block, x)
                else:
                    x = block(x)
            
            # Final layer norm
            x = self.ln_f(x)
            
            # Language modeling head
            logits = self.lm_head(x)
            
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
            return logits, loss
            
        except Exception as e:
            raise RuntimeError(f"Forward pass failed: {str(e)}")
    
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate text tokens autoregressively with better error handling"""
        if max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")
        
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        
        for token_idx in range(max_new_tokens):
            if idx.size(1) == 0:
                raise ValueError("Input sequence is empty. Please provide at least one token.")
            
            # Crop idx to the last block_size tokens
            idx_cond = idx if idx.size(1) <= self.block_size else idx[:, -self.block_size:]
            
            try:
                # Forward pass
                logits, _ = self(idx_cond)
                
                # Pluck the logits at the final step and scale by temperature
                logits = logits[:, -1, :] / temperature
                
                # Optionally crop the logits to only the top k options
                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                
                # Apply softmax to convert to probabilities
                probs = F.softmax(logits, dim=-1)
                
                # Sample from the distribution
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # Append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1)
                
            except Exception as e:
                raise RuntimeError(f"Generation failed at token {token_idx}: {str(e)}")
        
        return idx
    
    def crop_block_size(self, block_size):
        """Adjust the model's internal block_size if needed"""
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        
        self.config.block_size = block_size
        self.block_size = block_size

def configure_optimizers_self_attn(model, weight_decay, learning_rate, betas, device_type):
    """
    Creates and returns an AdamW optimizer for the self-attention model's parameters,
    following GPT-2 style parameter grouping with weight decay.
    """
    # Input validation
    if weight_decay < 0:
        raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")
    if learning_rate <= 0:
        raise ValueError(f"learning_rate must be positive, got {learning_rate}")
    if len(betas) != 2 or not all(0 <= b < 1 for b in betas):
        raise ValueError(f"betas must be two values in [0, 1), got {betas}")
    
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
    
    if len(inter_params) > 0:
        raise RuntimeError(f"parameters {inter_params} made it into both decay/no_decay sets!")
    
    missing_params = param_dict.keys() - union_params
    if len(missing_params) > 0:
        raise RuntimeError(f"parameters {missing_params} were not separated into either decay/no_decay set!")
    
    # Create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    
    try:
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer
    except Exception as e:
        raise RuntimeError(f"Failed to create optimizer: {str(e)}") 