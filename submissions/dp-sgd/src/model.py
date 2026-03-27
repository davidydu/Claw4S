"""Two-layer MLP for DP-SGD classification experiments."""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation.

    Architecture: input -> Linear(hidden) -> ReLU -> Linear(n_classes)
    """

    def __init__(self, n_features: int = 10, n_hidden: int = 64, n_classes: int = 5):
        super().__init__()
        self.fc1 = nn.Linear(n_features, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def create_model(
    n_features: int = 10,
    n_hidden: int = 64,
    n_classes: int = 5,
    seed: int = 42,
) -> MLP:
    """Create and initialize an MLP with a fixed seed.

    Args:
        n_features: Input dimension.
        n_hidden: Hidden layer width.
        n_classes: Number of output classes.
        seed: Random seed for weight initialization.

    Returns:
        Initialized MLP model.
    """
    torch.manual_seed(seed)
    model = MLP(n_features=n_features, n_hidden=n_hidden, n_classes=n_classes)
    return model
