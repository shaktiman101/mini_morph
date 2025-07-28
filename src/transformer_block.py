import torch
import torch.nn as nn

from src.attention_block.mha.mha_torch_fa import MHA
from src.ffn.swiglu import SwiGLU_FFN
from src.pos_embed.rope import RotaryEmbedding


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MHA(cfg)
        self.swiglu_ffn = SwiGLU_FFN(cfg['emb_dim'], cfg['hidden_dim'])  # Example hidden dimension as 4 times the output dimension
        self.norm1 = nn.LayerNorm(cfg['emb_dim'])
        self.norm2 = nn.LayerNorm(cfg['emb_dim'])
        
    def forward(self, X, mask, cos, sin):
        # encoding
        # embedding
        # positional encoding
        
        # Attention Block
        skip_connection = X
        X = self.norm1(X)
        X = self.mha(X, mask, cos, sin)
        X = X + skip_connection
        
        # FFN Block
        skip_connection = X
        X = self.norm2(X)
        X = self.swiglu_ffn(X)
        X = X + skip_connection
        
        return X