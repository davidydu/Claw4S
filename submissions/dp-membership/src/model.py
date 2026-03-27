"""Two-layer MLP for classification.

Simple architecture used as both target and shadow models
in the membership inference experiment.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation.

    Args:
        input_dim: Number of input features.
        hidden_dim: Number of hidden units.
        num_classes: Number of output classes.
    """

    def __init__(self, input_dim: int = 10, hidden_dim: int = 64, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
