import torch.nn as nn

from src.attention_block.gqa import GroupedQueryAttention
from src.ffn.swiglu import SwiGLU_FFN


class Llama3_2TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gqa = GroupedQueryAttention(cfg)
        self.swiglu_ffn = SwiGLU_FFN(cfg['emb_dim'], cfg['hidden_dim'])
        self.norm1 = nn.LayerNorm(cfg['emb_dim'])
        self.norm2 = nn.LayerNorm(cfg['emb_dim'])
        
    def forward(self, X, mask, cos, sin):
        # encoding - encode tokens in dataset itself
        # embedding - embed tokens outside transformer block
        # positional encoding - apply positional encoding in transformer block
        
        # Attention Block
        skip_connection = X
        X = self.norm1(X)
        X = self.gqa(X, mask, cos, sin) # apply RoPE during attention module
        X = X + skip_connection
        
        # FFN Block
        skip_connection = X
        X = self.norm2(X)
        X = self.swiglu_ffn(X)
        X = X + skip_connection
        
        return X