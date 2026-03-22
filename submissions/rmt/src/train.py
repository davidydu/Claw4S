"""Training loop for tiny MLPs."""

import torch
import torch.nn as nn
from src.model import TinyMLP


def train_model(
    model: TinyMLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    task: str = "classification",
    epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
    batch_size: int = 512,
    verbose: bool = True,
) -> dict:
    """Train a TinyMLP and return training history.

    Args:
        model: TinyMLP instance.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        task: 'classification' or 'regression'.
        epochs: Number of training epochs.
        lr: Learning rate.
        seed: Random seed.
        batch_size: Mini-batch size.
        verbose: Print progress every 100 epochs.

    Returns:
        Dict with loss_history, final_loss, and task-specific metric
        (final_accuracy for classification, final_mse for regression).
    """
    torch.manual_seed(seed)

    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task: {task}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    n_train = X_train.shape[0]

    for epoch in range(epochs):
        model.train()

        # Mini-batch training
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            X_batch = X_train[idx]
            y_batch = y_train[idx]

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if verbose and (epoch + 1) % 100 == 0:
            print(f"    Epoch {epoch + 1}/{epochs}, loss={avg_loss:.4f}")

    # Evaluate
    model.eval()
    with torch.no_grad():
        test_output = model(X_test)
        test_loss = criterion(test_output, y_test).item()

        result = {
            "loss_history": loss_history,
            "final_loss": test_loss,
        }

        if task == "classification":
            preds = test_output.argmax(dim=1)
            accuracy = (preds == y_test).float().mean().item()
            result["final_accuracy"] = accuracy
            if verbose:
                print(f"    Test accuracy: {accuracy:.4f}")
        else:
            result["final_mse"] = test_loss
            if verbose:
                print(f"    Test MSE: {test_loss:.6f}")

    return result
