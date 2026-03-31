"""
MLP models of varying widths for adversarial robustness experiments.

All models are 2-layer ReLU MLPs: input -> hidden -> hidden -> output.
"""

import torch
import torch.nn as nn


class TwoLayerMLP(nn.Module):
    """A 2-hidden-layer ReLU MLP for binary classification.

    Architecture: Linear(in, hidden) -> ReLU -> Linear(hidden, hidden) -> ReLU -> Linear(hidden, 2)

    Args:
        input_dim: Dimensionality of input features.
        hidden_width: Number of neurons in each hidden layer.
    """

    def __init__(self, input_dim: int = 2, hidden_width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, 2),
        )
        self.input_dim = input_dim
        self.hidden_width = hidden_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits of shape (batch, 2)."""
        return self.net(x)

    def param_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(hidden_width: int, input_dim: int = 2, seed: int = 42) -> TwoLayerMLP:
    """Build and deterministically initialize a TwoLayerMLP.

    Args:
        hidden_width: Number of neurons in each hidden layer.
        input_dim: Dimensionality of input features.
        seed: Random seed for weight initialization.

    Returns:
        Initialized TwoLayerMLP model.
    """
    torch.manual_seed(seed)
    model = TwoLayerMLP(input_dim=input_dim, hidden_width=hidden_width)
    return model


HIDDEN_WIDTHS = [16, 32, 64, 128, 256, 512]
