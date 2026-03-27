"""Model training utilities."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import MLP


def train_model(
    model: MLP,
    dataset: TensorDataset,
    lr: float = 0.01,
    epochs: int = 50,
    batch_size: int = 64,
    seed: int = 42,
) -> dict:
    """Train an MLP classifier on the given dataset.

    Args:
        model: MLP model to train.
        dataset: TensorDataset with (X, y) tensors.
        lr: Learning rate.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        seed: Random seed for DataLoader shuffling.

    Returns:
        Dict with 'final_loss' and 'final_accuracy'.
    """
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    final_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        final_loss = epoch_loss / max(n_batches, 1)

    # Compute final accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            preds = model(X_batch).argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return {
        "final_loss": final_loss,
        "final_accuracy": correct / max(total, 1),
    }
