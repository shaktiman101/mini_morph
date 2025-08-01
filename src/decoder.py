import torch
import torch.nn as nn

from src.transformer_block.gpt2 import GPT2TransformerBlock
from src.transformer_block.llama import Llama3_2TransformerBlock
from src.transformer_block.qwen import Qwen3TransformerBlock
from src.normalization.rms_norm import RMSNorm, compute_rope_params


def load_transformer_block(cfg):
    match cfg['model_name']:
        case 'gpt2':
            return GPT2TransformerBlock(cfg)
        case 'llama3_2':
            return Llama3_2TransformerBlock(cfg)
        case 'qwen3':
            return Qwen3TransformerBlock(cfg)
        case _:
            raise ValueError(f"Unsupported model name: {cfg['model_name']}")


class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model_name = cfg['model_name']
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        
        TransformerBlock = load_transformer_block(cfg)
        self.trf_blocks = nn.ModuleList(
            [TransformerBlock for _ in range(cfg['n_layers'])]
        )
        
        self.final_norm = RMSNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias=False, dtype=cfg['dtype'])
        
        # revisit
        if cfg["head_dim"] is None:
            head_dim = cfg["emb_dim"] // cfg["n_heads"]
        else:
            head_dim = cfg["head_dim"]
        cos, sin = compute_rope_params(
            head_dim=head_dim,
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"]
        )
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.cfg = cfg
        
    def forward(self, X):
        # embedding
        X = self.tok_emb(X)
        
        num_tokens = X.shape[1]
        
        # # positional encoding
        # X = X * self.cos + X * self.sin
        mask = torch.triu(torch.ones(num_tokens, num_tokens, device=X.device, dtype=torch.bool), diagonal=1)
        
        # transformer blocks
        for block in self.trf_blocks:
            X = block(X, mask, self.cos, self.sin)
        
        # final normalization
        X = self.final_norm(X)

        # output head
        logits = self.out_head(X.to(self.cfg['dtype']))
        
        return logits