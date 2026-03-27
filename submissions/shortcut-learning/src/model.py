"""Two-layer MLP for binary classification."""

import torch
import torch.nn as nn


class ShortcutMLP(nn.Module):
    """Simple 2-layer MLP: input -> hidden (ReLU) -> hidden (ReLU) -> output.

    Args:
        input_dim: Number of input features.
        hidden_dim: Width of each hidden layer.
    """

    def __init__(self, input_dim: int = 11, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def create_model(input_dim: int = 11, hidden_dim: int = 64) -> ShortcutMLP:
    """Factory function to create a ShortcutMLP.

    Args:
        input_dim: Number of input features (genuine + shortcut).
        hidden_dim: Width of each hidden layer.

    Returns:
        Initialized ShortcutMLP model.
    """
    return ShortcutMLP(input_dim=input_dim, hidden_dim=hidden_dim)
