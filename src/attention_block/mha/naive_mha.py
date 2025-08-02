import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, dim_in, dim_out, max_len, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.W_query = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_key = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.W_value = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(max_len, max_len), diagonal=1))
        
    def forward(self, X):
        batch, num_tokens, dim_in = X.shape
        
        queries = self.W_query(X)
        keys = self.W_key(X)
        values = self.W_value(X)
        
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf # type: ignore
        )
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
    
    
class NaiveMHA(nn.Module):
    """Naive Multi-Head Attention implementation.
    Applies multiple causal attention heads sequentially (for loop)and concatenates their outputs."""
    def __init__(self, num_heads, dim_in, dim_out, max_len, qkv_bias=False, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList(
            [CausalAttention(dim_in, dim_out, max_len, qkv_bias, dropout)
             for head in range(num_heads)]
        )
        self.out_proj = nn.Linear(num_heads*dim_out, dim_out*num_heads)
        
    def forward(self, X):
        context_vec = torch.cat([head(X) for head in self.heads], dim=-1)
        return self.out_proj(context_vec)