import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-6, bias=False):
        super().__init__()
        self.eps = eps
        self.embed_dim = embed_dim
        self.scale = nn.Parameter(torch.ones(self.embed_dim))
        self.shift = nn.Parameter(torch.zeros(self.embed_dim)) if bias else None
        
    def forward(self, X):
        input_dtype = X.dtype
        X = X.to(torch.float32)
        
        var = X.pow(2).mean(dim=-1, keepdim=True)
        norm_X = X / (var + self.eps).sqrt()
        norm_X = norm_X * self.scale
        
        if self.shift is not None:
            norm_X = norm_X + self.shift
        return norm_X.to(input_dtype)
    
    
def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):
    assert head_dim % 2 == 0, "Embedding dimension must be even"

    # Compute the inverse frequencies
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim, 2, dtype=dtype)[: (head_dim // 2)].float() / head_dim))

    # Generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # Compute the angles
    angles = positions[:, None] * inv_freq[None, :]  # Shape: (context_length, head_dim // 2)

    # Expand angles to match the head_dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # Precompute sine and cosine
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):
    # x: (batch_size, num_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head dimension must be even"

    # Split x into first half and second half
    x1 = x[..., : head_dim // 2]  # First half
    x2 = x[..., head_dim // 2 :]  # Second half

    # Adjust sin and cos shapes
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, seq_len, head_dim)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # Apply the rotary transformation
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # It's ok to use lower-precision after applying cos and sin rotation
    return x_rotated.to(dtype=x.dtype)