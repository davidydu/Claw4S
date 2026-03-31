"""MLP models for calibration experiments.

Provides 2-layer MLPs of varying widths to study how overparameterization
affects calibration under distribution shift.
"""

import torch
import torch.nn as nn
from typing import Tuple


class TwoLayerMLP(nn.Module):
    """Simple 2-layer MLP with ReLU activation.

    Architecture: input -> Linear(hidden) -> ReLU -> Linear(n_classes) -> softmax

    Args:
        input_dim: Number of input features.
        hidden_dim: Width of hidden layer.
        n_classes: Number of output classes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits."""
        return self.net(x)


def train_model(model: TwoLayerMLP,
                X_train: torch.Tensor,
                y_train: torch.Tensor,
                lr: float = 0.01,
                epochs: int = 200,
                weight_decay: float = 0.0) -> list[float]:
    """Train model to convergence with cross-entropy loss.

    Args:
        model: The MLP to train.
        X_train: Training features, shape (n_samples, n_features).
        y_train: Training labels, shape (n_samples,).
        lr: Learning rate.
        epochs: Number of training epochs.
        weight_decay: L2 regularization strength.

    Returns:
        List of per-epoch training losses.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    losses = []

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def predict_proba(model: TwoLayerMLP,
                  X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Get predicted probabilities and class predictions.

    Args:
        model: Trained MLP.
        X: Input features.

    Returns:
        Tuple of (probabilities, predicted_classes).
    """
    model.eval()
    with torch.no_grad():
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
    return probs, preds
