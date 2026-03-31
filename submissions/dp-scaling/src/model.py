"""Two-layer MLP for classification experiments.

Provides a simple parameterizable MLP and a function to count trainable
parameters, used to study how model size affects loss under DP-SGD.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation.

    Architecture: Linear(in, hidden) -> ReLU -> Linear(hidden, out)

    Args:
        n_features: Number of input features.
        hidden_size: Number of hidden units (controls model size).
        n_classes: Number of output classes.
    """

    def __init__(self, n_features: int, hidden_size: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.fc2(self.relu(self.fc1(x)))


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in a model.

    Args:
        model: PyTorch module.

    Returns:
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
