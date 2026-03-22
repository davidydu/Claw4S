"""Training loop for lottery ticket experiments.

Supports both classification (modular arithmetic) and regression tasks.
"""

import torch
import torch.nn as nn
from src.model import TwoLayerMLP
from src.pruning import apply_masks


def train_classification(
    model: TwoLayerMLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    masks: dict,
    max_epochs: int = 3000,
    lr: float = 1e-2,
    patience: int = 200,
) -> dict:
    """Train model on classification task (modular arithmetic).

    Args:
        model: The MLP to train.
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        masks: Pruning masks to re-apply after each step.
        max_epochs: Maximum training epochs.
        lr: Learning rate.
        patience: Stop if test accuracy doesn't improve for this many epochs.

    Returns:
        Dictionary with train_acc, test_acc, train_loss, epochs_trained.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    epochs_without_improvement = 0
    final_train_acc = 0.0
    final_test_acc = 0.0
    final_train_loss = 0.0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

        # Re-apply pruning masks
        if masks:
            apply_masks(model, masks)

        # Evaluate
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).argmax(dim=1)
            train_acc = (train_preds == y_train).float().mean().item()

            test_preds = model(X_test).argmax(dim=1)
            test_acc = (test_preds == y_test).float().mean().item()

        final_train_acc = train_acc
        final_test_acc = test_acc
        final_train_loss = loss.item()

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        # Early stop if perfect
        if test_acc >= 0.999:
            break

    return {
        "train_acc": final_train_acc,
        "test_acc": final_test_acc,
        "train_loss": final_train_loss,
        "epochs_trained": epoch + 1,
    }


def train_regression(
    model: TwoLayerMLP,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    masks: dict,
    max_epochs: int = 3000,
    lr: float = 1e-3,
    patience: int = 200,
) -> dict:
    """Train model on regression task.

    Uses R-squared as the accuracy metric.

    Args:
        model: The MLP to train.
        X_train, y_train: Training data and targets.
        X_test, y_test: Test data and targets.
        masks: Pruning masks to re-apply after each step.
        max_epochs: Maximum training epochs.
        lr: Learning rate.
        patience: Stop if test R^2 doesn't improve for this many epochs.

    Returns:
        Dictionary with train_r2, test_r2, train_loss, epochs_trained.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_test_r2 = -float("inf")
    epochs_without_improvement = 0
    final_train_r2 = 0.0
    final_test_r2 = 0.0
    final_train_loss = 0.0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_train).squeeze()
        loss = criterion(preds, y_train)
        loss.backward()
        optimizer.step()

        # Re-apply pruning masks
        if masks:
            apply_masks(model, masks)

        # Evaluate with R-squared
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train).squeeze()
            test_preds = model(X_test).squeeze()

            train_r2 = _r_squared(y_train, train_preds)
            test_r2 = _r_squared(y_test, test_preds)

        final_train_r2 = train_r2
        final_test_r2 = test_r2
        final_train_loss = loss.item()

        if test_r2 > best_test_r2:
            best_test_r2 = test_r2
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

        # Early stop if very good fit
        if test_r2 >= 0.999:
            break

    return {
        "train_r2": final_train_r2,
        "test_r2": final_test_r2,
        "train_loss": final_train_loss,
        "epochs_trained": epoch + 1,
    }


def _r_squared(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R-squared (coefficient of determination).

    R^2 = 1 - SS_res / SS_tot
    """
    ss_res = ((y_true - y_pred) ** 2).sum().item()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum().item()
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot
