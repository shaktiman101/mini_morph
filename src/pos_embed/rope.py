# revisit
import torch
import torch.nn as nn


def rotate_half(x):
    # splits into two halves and applies [x2, -x1]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def apply_rope(x, sin, cos):
    # x: [batch, seq_len, num_heads, head_dim]
    return (x * cos) + (rotate_half(x) * sin)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # duplicate for sin and cos
        sin = emb.sin()[None, :, None, :]  # [1, seq_len, 1, dim]
        cos = emb.cos()[None, :, None, :]
        return sin, cos
