import torch
import torch.nn as nn


class MHA(nn.Module):
    def __init__(self, dim_in, dim_out, max_len, num_heads, qkv_bias=False, dropout=0.0):
        super().__init__()
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(max_len, max_len), diagonal=1))
        
    def forward(self, X):
        batch, num_tokens, dim_in = X.shape
        
        queries = self.W_query(X)
        keys = self.W_key(X)
        values = self.W_value(X)
        
        # Reshape QKV matrices for mult heads
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)
        
        queries = queries.transpose(1, 2)  # (batch, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        # Calculate attention scores
        attn_scores = queries @ keys.transpose(2, 3)    # revisit

        # Mask
        mask = self.mask.bool()[:num_tokens, :num_tokens]   # revisit
        attn_scores.masked_fill_(mask, -torch.inf)
        
        # Attention weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5,  dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Scale attention weights
        context_vec = (attn_weights @ values).transpose(1, 2)  # (batch, num_heads, num_tokens, head_dim)   # revisit
        
        context_vec = context_vec.contiguous().view(batch, num_tokens, self.dim_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec