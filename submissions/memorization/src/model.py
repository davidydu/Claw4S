# src/model.py
"""Parameterized MLP for memorization experiments."""

import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP: Linear -> ReLU -> Linear.

    Architecture: input_dim -> hidden_dim -> num_classes
    Total params: input_dim * hidden_dim + hidden_dim + hidden_dim * num_classes + num_classes
               = hidden_dim * (input_dim + num_classes) + (hidden_dim + num_classes)
    """

    def __init__(self, input_dim: int = 20, hidden_dim: int = 64, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
