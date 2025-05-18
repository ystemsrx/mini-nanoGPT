# src/gpt_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

class GPTConfig:
    def __init__(
        self, 
        vocab_size, 
        block_size, 
        n_layer, 
        n_head, 
        n_embd, 
        dropout, 
        bias=False
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias

class GPT(nn.Module):
    """
    A minimal GPT-like model consisting of an embedding layer and a linear layer. 
    This is primarily for demonstration purposes.
    """
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.block_size = config.block_size
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=config.bias)

    def forward(self, idx, targets=None):
        # idx shape: (batch, time)
        b, t = idx.size()
        # Convert token indices to embeddings
        token_emb = self.token_embedding_table(idx)
        # Project embeddings onto vocab space
        logits = self.lm_head(token_emb)

        loss = None
        if targets is not None:
            # Flatten logits/targets for cross entropy
            logits_view = logits.view(b * t, -1)
            targets_view = targets.view(b * t)
            loss = F.cross_entropy(logits_view, targets_view)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generates tokens one at a time, appending each new token to 'idx'. 
        The process continues until 'max_new_tokens' tokens have been added.
        """
        for _ in range(max_new_tokens):
            if idx.size(1) == 0:
                raise ValueError("Input sequence is empty. Please provide at least one token.")
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            # Scale logits by temperature
            logits = logits[:, -1, :] / temperature
            # If top_k is set, zero out logits not in top_k
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                top_value = v[:, -1].unsqueeze(-1)
                logits[logits < top_value] = -float('Inf')
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append new token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def crop_block_size(self, block_size):
        """
        Adjust the model's internal block_size if needed. 
        Useful when resuming training with a different context length.
        """
        self.config.block_size = block_size
        self.block_size = block_size

# ————————————————————————————————————————————————————————————————————————————
# Optimizer Configuration
# ————————————————————————————————————————————————————————————————————————————

def configure_optimizers(model, weight_decay, learning_rate, betas, device_type):
    """
    Creates and returns an AdamW optimizer for the model's parameters, 
    ignoring those that do not require gradients.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    return optimizer
