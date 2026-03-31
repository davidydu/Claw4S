"""Training loop for MLPs on synthetic classification data."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def train_model(
    model: nn.Module,
    train_ds: TensorDataset,
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    seed: int = 42,
) -> list[float]:
    """Train *model* on *train_ds* and return per-epoch training losses.

    Uses SGD with no momentum (simplest baseline).  Deterministic via manual
    seed on the data-loader generator.
    """
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=g
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    epoch_losses: list[float] = []

    for _ in range(n_epochs):
        running_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        epoch_losses.append(running_loss / max(n_batches, 1))

    return epoch_losses


@torch.no_grad()
def evaluate(model: nn.Module, ds: TensorDataset) -> float:
    """Return accuracy (0-1) of *model* on dataset *ds*."""
    model.eval()
    X_all = ds.tensors[0]
    y_all = ds.tensors[1]
    logits = model(X_all)
    preds = logits.argmax(dim=1)
    acc = (preds == y_all).float().mean().item()
    return acc
