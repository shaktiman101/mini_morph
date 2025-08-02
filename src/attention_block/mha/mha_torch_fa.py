import torch
import torch.nn as nn

from src.normalization.rms_norm import apply_rope


class MHA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dim_in=cfg['emb_dim']
        dim_out=cfg['emb_dim']
        max_len=cfg['max_len']
        num_heads=cfg['n_heads']
        qkv_bias=cfg['qkv_bias']
        dropout=cfg['dropout']
        training=cfg['training']
        assert dim_out % num_heads == 0, "dim_out must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim_out // num_heads
        self.dim_in = dim_in
        self.dim_out = dim_out
        
        self.qkv = nn.Linear(dim_in, 3*dim_out, bias=qkv_bias)
        self.out_proj = nn.Linear(dim_out, dim_out)
        self.dropout = dropout
        self.training = training
        
    def forward(self, X, mask, cos, sin):
        batch, num_tokens, dim_in = X.shape
        
        # (b, num_tokens, embed_dim) --> (b, num_tokens, 3 * embed_dim)
        qkv = self.qkv(X)
        
        # (b, num_tokens, 3 * embed_dim) --> (b, num_tokens, 3, num_heads, head_dim)
        qkv = qkv.view(batch, num_tokens, 3, self.num_heads, self.head_dim)
        
        # (b, num_tokens, 3, num_heads, head_dim) --> (3, b, num_heads, num_tokens, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        # (3, b, num_heads, num_tokens, head_dim) -> 3 times (b, num_head, num_tokens, head_dim)
        queries, keys, values = qkv.unbind(0)
        
        ####################
        # Apply RoPE
        if False:
            queries = apply_rope(queries, cos, sin)
            keys = apply_rope(keys, cos, sin)

            # Expand K and V to match number of heads
            keys = keys.repeat_interleave(self.group_size, dim=1)
            values = values.repeat_interleave(self.group_size, dim=1)
        #################
        
        use_dropout = 0. if not self.training else self.dropout

        context_vec = nn.functional.scaled_dot_product_attention(
            queries, keys, values, attn_mask=None, dropout_p=use_dropout, is_causal=True)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.transpose(1, 2).contiguous().view(batch, num_tokens, self.dim_out)
        context_vec = self.out_proj(context_vec)
        
        return context_vec
