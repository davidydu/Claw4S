"""Training loop with weight snapshot saving for Benford's Law experiments."""

import copy
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def train_model(
    model,
    X_train,
    y_train,
    task_type,
    epochs=5000,
    lr=1e-3,
    snapshot_epochs=None,
    seed=42,
    batch_size=512,
    X_test=None,
    y_test=None,
    log_every=None,
    log_prefix="",
):
    """Train a model and save weight snapshots at specified epochs.

    Args:
        model: nn.Module to train.
        X_train: Training inputs.
        y_train: Training targets.
        task_type: "classification" or "regression".
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        snapshot_epochs: List of epochs at which to save weight snapshots.
            Defaults to [0, 100, 500, 1000, 2000, 5000].
        seed: Random seed.
        batch_size: Mini-batch size.
        X_test: Optional test inputs for tracking test loss.
        y_test: Optional test targets for tracking test loss.
        log_every: Optional epoch interval for progress logging.
            If None, disables periodic logging.
        log_prefix: Optional prefix prepended to progress log lines.

    Returns:
        Tuple of (snapshots, history):
        - snapshots: dict of {epoch: state_dict_copy}
        - history: dict with keys "train_loss", "test_loss" (lists of floats)
    """
    if snapshot_epochs is None:
        snapshot_epochs = [0, 100, 500, 1000, 2000, 5000]

    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    snapshots = {}
    history = {"train_loss": [], "test_loss": []}

    # Snapshot at epoch 0 (before training)
    if 0 in snapshot_epochs:
        snapshots[0] = copy.deepcopy(model.state_dict())

    n_samples = X_train.shape[0]

    for epoch in range(1, epochs + 1):
        model.train()

        # Mini-batch training (seeded generator for reproducibility)
        perm = torch.randperm(n_samples, generator=g)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_samples, batch_size):
            batch_idx = perm[i : i + batch_size]
            X_batch = X_train[batch_idx]
            y_batch = y_train[batch_idx]

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        history["train_loss"].append(avg_loss)

        # Test loss
        if X_test is not None and y_test is not None:
            model.eval()
            with torch.no_grad():
                test_output = model(X_test)
                test_loss = criterion(test_output, y_test).item()
            history["test_loss"].append(test_loss)

        # Save snapshot
        if epoch in snapshot_epochs:
            snapshots[epoch] = copy.deepcopy(model.state_dict())

        # Optional progress logging for long runs
        if log_every and (epoch % log_every == 0 or epoch == epochs):
            message = f"epoch {epoch}/{epochs} train_loss={avg_loss:.6f}"
            if history["test_loss"]:
                message += f" test_loss={history['test_loss'][-1]:.6f}"
            if log_prefix:
                message = f"{log_prefix} {message}"
            print(message, flush=True)

    return snapshots, history
