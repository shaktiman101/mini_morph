import torch
import torch.nn as nn


class SiLU_FFN(nn.Module):
    """SiLU activation function (Sigmoid Linear Unit)"""
    def __init__(self, embed_dim, hidden_dim, bias=False):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.activation = nn.SiLU()

    def forward(self, X):
        return self.fc2(self.activation(self.fc1(X)))