"""Training loop with weight decay regularization."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.model import create_model


def train_model(
    train_dataset: TensorDataset,
    input_dim: int = 11,
    hidden_dim: int = 64,
    weight_decay: float = 0.0,
    seed: int = 42,
    epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 128,
) -> nn.Module:
    """Train a ShortcutMLP on the given dataset.

    Args:
        train_dataset: TensorDataset of (X, y) pairs.
        input_dim: Number of input features.
        hidden_dim: Width of hidden layers.
        weight_decay: L2 regularization strength for Adam optimizer.
        seed: Random seed for reproducibility.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.

    Returns:
        Trained model in eval mode.
    """
    torch.manual_seed(seed)

    model = create_model(input_dim=input_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model.train()
    for _epoch in range(epochs):
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    model.eval()
    return model


def evaluate_accuracy(model: nn.Module, dataset: TensorDataset) -> float:
    """Compute classification accuracy on a dataset.

    Args:
        model: Trained model in eval mode.
        dataset: TensorDataset of (X, y) pairs.

    Returns:
        Accuracy as a float in [0, 1].
    """
    loader = DataLoader(dataset, batch_size=512, shuffle=False)
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total
