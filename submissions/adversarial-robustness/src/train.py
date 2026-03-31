"""
Training loop for MLP models on synthetic classification datasets.

Trains to convergence with early stopping based on loss plateau.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train_model(model: nn.Module, train_loader: DataLoader,
                max_epochs: int = 2000, lr: float = 1e-3,
                patience: int = 50, min_delta: float = 1e-5,
                seed: int = 42, verbose: bool = False) -> dict:
    """Train a model to convergence with early stopping.

    Args:
        model: PyTorch model to train.
        train_loader: Training data loader.
        max_epochs: Maximum number of training epochs.
        lr: Learning rate for Adam optimizer.
        patience: Number of epochs with no improvement before stopping.
        min_delta: Minimum loss decrease to count as improvement.
        seed: Random seed for reproducibility.
        verbose: If True, print progress every 100 epochs.

    Returns:
        Dictionary with keys: final_epoch, final_loss, loss_history.
    """
    torch.manual_seed(seed)
    device = torch.device("cpu")
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    epochs_without_improvement = 0
    loss_history: list[float] = []

    for epoch in range(1, max_epochs + 1):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        loss_history.append(avg_loss)

        if avg_loss < best_loss - min_delta:
            best_loss = avg_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if verbose and epoch % 100 == 0:
            print(f"  Epoch {epoch:4d}: loss={avg_loss:.6f}")

        if epochs_without_improvement >= patience:
            if verbose:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    return {
        "final_epoch": epoch,
        "final_loss": avg_loss,
        "loss_history": loss_history,
    }


def evaluate_clean(model: nn.Module, X_test: torch.Tensor,
                   y_test: torch.Tensor) -> float:
    """Evaluate clean (non-adversarial) accuracy.

    Args:
        model: Trained model.
        X_test: Test inputs, shape (N, D).
        y_test: Test labels, shape (N,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    with torch.no_grad():
        logits = model(X_test)
        preds = logits.argmax(dim=1)
        accuracy = (preds == y_test).float().mean().item()
    return accuracy
