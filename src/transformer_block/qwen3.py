import torch.nn as nn

from src.attention_block.gqa import GroupedQueryAttention
from src.ffn.swiglu import SwiGLU_FFN


class QwenTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gqa = GroupedQueryAttention(cfg)
        self.swiglu_ffn = SwiGLU_FFN(cfg['emb_dim'], cfg['hidden_dim'])
        self.norm1 = nn.LayerNorm(cfg['emb_dim'])
        self.norm2 = nn.LayerNorm(cfg['emb_dim'])
        
    def forward(self, X, mask, cos, sin):        
        # Attention Block
        skip_connection = X
        X = self.norm1(X)
        # revisit
        # apply Q/K RMSNorm
        X = self.gqa(X, mask, cos, sin) # apply RoPE during attention module
        X = X + skip_connection
        
        # FFN Block
        skip_connection = X
        X = self.norm2(X)
        X = self.swiglu_ffn(X)
        X = X + skip_connection
        
        return X