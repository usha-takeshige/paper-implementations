import torch
import torch.nn as nn
from torch import Tensor


class Generator(nn.Module):
    def __init__(self, latent_dim: int, null_dim: int, hidden_dims: list[int] | None = None):
        super().__init__()
        self.latent_dim = latent_dim
        self.null_dim = null_dim
        self.hidden_dims = hidden_dims or [64, 64]

        layers: list[nn.Module] = []
        in_dim = latent_dim
        for h in self.hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.Tanh()])
            in_dim = h
        layers.append(nn.Linear(in_dim, null_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)
