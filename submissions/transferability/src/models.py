"""MLP model definitions with configurable width and depth."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-layer perceptron with configurable width and depth.

    Args:
        input_dim: Number of input features.
        n_classes: Number of output classes.
        hidden_width: Width of each hidden layer.
        n_hidden_layers: Number of hidden layers (default 2).
    """

    def __init__(
        self,
        input_dim: int,
        n_classes: int,
        hidden_width: int,
        n_hidden_layers: int = 2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.hidden_width = hidden_width
        self.n_hidden_layers = n_hidden_layers

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(nn.ReLU())
            in_dim = hidden_width
        layers.append(nn.Linear(in_dim, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def param_count(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
