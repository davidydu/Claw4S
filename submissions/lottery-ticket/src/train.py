"""Training loop for lottery ticket experiments.

Supports both classification (modular arithmetic) and regression tasks.
"""

import copy

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
    validation_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Train model on classification task (modular arithmetic).

    Args:
        model: The MLP to train.
        X_train, y_train: Training data and labels.
        X_test, y_test: Test data and labels.
        masks: Pruning masks to re-apply after each step.
        max_epochs: Maximum training epochs.
        lr: Learning rate.
        patience: Stop if validation accuracy does not improve for this many epochs.
        validation_fraction: Fraction of the training set reserved for validation.
        seed: Random seed for the deterministic train/validation split.

    Returns:
        Dictionary with train/validation/test metrics and training metadata.
    """
    X_opt, y_opt, X_val, y_val = _split_train_validation(
        X_train, y_train, validation_fraction=validation_fraction, seed=seed
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = -float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(X_opt)
        loss = criterion(logits, y_opt)
        loss.backward()
        optimizer.step()

        if masks:
            apply_masks(model, masks)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == y_val).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break
        if val_acc >= 0.999:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_logits = model(X_train)
        val_logits = model(X_val)
        test_logits = model(X_test)

        train_preds = train_logits.argmax(dim=1)
        val_preds = val_logits.argmax(dim=1)
        test_preds = test_logits.argmax(dim=1)

        final_train_acc = (train_preds == y_train).float().mean().item()
        final_val_acc = (val_preds == y_val).float().mean().item()
        final_test_acc = (test_preds == y_test).float().mean().item()
        final_train_loss = criterion(train_logits, y_train).item()

    return {
        "train_acc": final_train_acc,
        "val_acc": final_val_acc,
        "test_acc": final_test_acc,
        "train_loss": final_train_loss,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
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
    validation_fraction: float = 0.1,
    seed: int = 42,
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
        patience: Stop if validation R^2 does not improve for this many epochs.
        validation_fraction: Fraction of the training set reserved for validation.
        seed: Random seed for the deterministic train/validation split.

    Returns:
        Dictionary with train/validation/test metrics and training metadata.
    """
    X_opt, y_opt, X_val, y_val = _split_train_validation(
        X_train, y_train, validation_fraction=validation_fraction, seed=seed
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_r2 = -float("inf")
    best_epoch = 0
    best_state = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        model.train()
        optimizer.zero_grad()
        preds = model(X_opt).squeeze()
        loss = criterion(preds, y_opt)
        loss.backward()
        optimizer.step()

        if masks:
            apply_masks(model, masks)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val).squeeze()
            val_r2 = _r_squared(y_val, val_preds)

        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break
        if val_r2 >= 0.999:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        train_preds = model(X_train).squeeze()
        val_preds = model(X_val).squeeze()
        test_preds = model(X_test).squeeze()

        final_train_r2 = _r_squared(y_train, train_preds)
        final_val_r2 = _r_squared(y_val, val_preds)
        final_test_r2 = _r_squared(y_test, test_preds)
        final_train_loss = criterion(train_preds, y_train).item()

    return {
        "train_r2": final_train_r2,
        "val_r2": final_val_r2,
        "test_r2": final_test_r2,
        "train_loss": final_train_loss,
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
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


def _split_train_validation(
    X: torch.Tensor,
    y: torch.Tensor,
    validation_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split tensors into optimization and validation subsets deterministically."""
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if len(X) < 2 or validation_fraction <= 0.0:
        return X, y, X, y

    n_val = max(1, int(round(len(X) * validation_fraction)))
    n_val = min(n_val, len(X) - 1)

    generator = torch.Generator()
    generator.manual_seed(seed)
    perm = torch.randperm(len(X), generator=generator)

    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]
