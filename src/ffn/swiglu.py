import torch
import torch.nn as nn


class SwiGLU_FFN(nn.Module):
    """SiLU activation function (Sigmoid Linear Unit)"""
    def __init__(self, embed_dim, hidden_dim, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc3 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, X):
        gate = self.fc1(X)
        act = self.activation(self.fc2(X))
        return self.fc3(gate * act)