# src/train.py
"""Training loop for memorization experiments."""

from dataclasses import dataclass, field

import torch
import torch.nn as nn


@dataclass
class TrainResult:
    """Results from a single training run."""
    final_train_acc: float
    final_train_loss: float
    convergence_epoch: int  # -1 if did not converge
    epochs_run: int
    loss_history: list[float] = field(default_factory=list)
    acc_history: list[float] = field(default_factory=list)


def compute_accuracy(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    """Compute classification accuracy."""
    model.eval()
    with torch.no_grad():
        logits = model(X)
        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
    return correct / len(y)


def train_model(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    max_epochs: int = 5000,
    lr: float = 0.001,
    convergence_loss: float = 1e-4,
    convergence_acc_epochs: int = 10,
    log_interval: int = 500,
    seed: int = 42,
) -> TrainResult:
    """Train model to memorize the dataset.

    Args:
        model: Neural network to train.
        X: Training features, shape (n, d).
        y: Training labels, shape (n,).
        max_epochs: Maximum number of training epochs.
        lr: Learning rate for Adam optimizer.
        convergence_loss: Stop if training loss falls below this value.
        convergence_acc_epochs: Stop if 100% accuracy held for this many consecutive epochs.
        log_interval: Print progress every N epochs.
        seed: Random seed for reproducibility.

    Returns:
        TrainResult with training metrics.
    """
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loss_history = []
    acc_history = []
    convergence_epoch = -1
    perfect_streak = 0

    for epoch in range(max_epochs):
        # Forward pass
        model.train()
        logits = model(X)
        loss = criterion(logits, y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        loss_val = loss.item()
        acc = compute_accuracy(model, X, y)
        loss_history.append(loss_val)
        acc_history.append(acc)

        # Check convergence
        if acc >= 1.0 - 1e-9:
            perfect_streak += 1
            if perfect_streak >= convergence_acc_epochs and convergence_epoch < 0:
                convergence_epoch = epoch - convergence_acc_epochs + 1
        else:
            perfect_streak = 0

        if loss_val < convergence_loss and convergence_epoch < 0:
            convergence_epoch = epoch

        # Early stopping: converged and loss is very low
        if convergence_epoch >= 0 and loss_val < convergence_loss:
            if log_interval > 0:
                print(f"    Converged at epoch {convergence_epoch} "
                      f"(loss={loss_val:.6f}, acc={acc:.4f})")
            break

        if log_interval > 0 and (epoch + 1) % log_interval == 0:
            print(f"    Epoch {epoch + 1}/{max_epochs}: "
                  f"loss={loss_val:.6f}, acc={acc:.4f}")

    final_acc = compute_accuracy(model, X, y)
    final_loss = loss_history[-1] if loss_history else float("inf")

    return TrainResult(
        final_train_acc=final_acc,
        final_train_loss=final_loss,
        convergence_epoch=convergence_epoch,
        epochs_run=len(loss_history),
        loss_history=loss_history,
        acc_history=acc_history,
    )
