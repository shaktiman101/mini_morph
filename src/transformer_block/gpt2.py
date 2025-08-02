import torch.nn as nn

from src.attention_block.mha.mha_torch_fa import MHA
from src.ffn.silu import SiLU_FFN


class GPT2TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MHA(cfg)
        self.swiglu_ffn = SiLU_FFN(cfg['emb_dim'], cfg['hidden_dim'])
        self.norm1 = nn.LayerNorm(cfg['emb_dim'])
        self.norm2 = nn.LayerNorm(cfg['emb_dim'])
        
    def forward(self, X, mask, cos, sin):        
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