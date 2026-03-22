"""Training routines for double descent experiments.

Provides:
- fit_random_features: Fit random features model (instant, least-squares).
- train_mlp: Train a neural network with gradient descent (for comparison).
"""

import torch
import torch.nn as nn

from src.model import RandomFeaturesModel


def fit_random_features(
    model: RandomFeaturesModel,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Fit a random features model and evaluate.

    Args:
        model: RandomFeaturesModel instance.
        X_train, y_train: Training data.
        X_test, y_test: Test data.

    Returns:
        Dict with train_loss, test_loss.
    """
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_mse = float(((train_pred - y_train) ** 2).mean())
    test_mse = float(((test_pred - y_test) ** 2).mean())

    return {
        "train_loss": train_mse,
        "test_loss": test_mse,
    }


def train_mlp(
    model: nn.Module,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    epochs: int = 4000,
    lr: float = 0.001,
    record_every: int = 0,
) -> dict:
    """Train an MLP and record train/test losses.

    Uses Adam optimizer with no weight decay (no regularization).

    Args:
        model: PyTorch model to train.
        X_train, y_train: Training data.
        X_test, y_test: Test data.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        record_every: If > 0, record losses every this many epochs.

    Returns:
        Dictionary with final losses and optional histories.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.MSELoss()

    train_losses = []
    test_losses = []
    epoch_list = []

    for epoch in range(1, epochs + 1):
        # Training step (full-batch)
        model.train()
        optimizer.zero_grad()
        pred_train = model(X_train)
        loss_train = criterion(pred_train, y_train)
        loss_train.backward()
        optimizer.step()

        # Record metrics
        if record_every > 0 and (epoch % record_every == 0 or epoch == 1):
            model.eval()
            with torch.no_grad():
                train_mse = criterion(model(X_train), y_train).item()
                test_mse = criterion(model(X_test), y_test).item()
            train_losses.append(train_mse)
            test_losses.append(test_mse)
            epoch_list.append(epoch)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        final_train = criterion(model(X_train), y_train).item()
        final_test = criterion(model(X_test), y_test).item()

    result = {
        "final_train_loss": final_train,
        "final_test_loss": final_test,
    }

    if record_every > 0:
        result["train_loss_history"] = train_losses
        result["test_loss_history"] = test_losses
        result["epoch_history"] = epoch_list

    return result
