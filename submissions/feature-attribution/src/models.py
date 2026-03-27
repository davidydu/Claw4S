"""MLP model definitions with configurable depth."""

import torch
import torch.nn as nn
from typing import List


class MLP(nn.Module):
    """Multi-layer perceptron with configurable hidden layers.

    Architecture: input -> [Linear+ReLU] * n_hidden -> Linear -> output
    All hidden layers have the same width.

    Args:
        n_features: Input dimensionality.
        n_classes: Number of output classes.
        n_hidden: Number of hidden layers (depth).
        width: Width of each hidden layer.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_hidden: int = 1,
        width: int = 64,
    ):
        super().__init__()
        layers: List[nn.Module] = []

        in_dim = n_features
        for _ in range(n_hidden):
            layers.append(nn.Linear(in_dim, width))
            layers.append(nn.ReLU())
            in_dim = width

        layers.append(nn.Linear(in_dim, n_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning raw logits."""
        return self.network(x)


def train_model(
    model: MLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    lr: float = 1e-3,
    epochs: int = 200,
    seed: int = 42,
) -> List[float]:
    """Train an MLP to convergence with Adam optimizer.

    Args:
        model: MLP instance.
        X_train: Training inputs (n, d).
        y_train: Training labels (n,).
        lr: Learning rate.
        epochs: Maximum training epochs.
        seed: Random seed for reproducibility.

    Returns:
        List of per-epoch loss values.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses
