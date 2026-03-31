"""Training loop for depth-width experiments."""

import time
import torch
import torch.nn as nn


def _compute_metric(
    output: torch.Tensor,
    target: torch.Tensor,
    task_type: str,
) -> tuple[float, str]:
    """Compute metric value and its display name for a task type."""
    if task_type == "classification":
        preds = output.argmax(dim=1)
        metric = (preds == target).float().mean().item()
        return metric, "accuracy"

    # Regression: R-squared
    output = output.squeeze(-1)
    ss_res = ((target - output) ** 2).sum().item()
    ss_tot = ((target - target.mean()) ** 2).sum().item()
    metric = 1.0 - ss_res / max(ss_tot, 1e-8)
    return metric, "r_squared"


def _split_train_validation(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    validation_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split training data into optimization-train and validation subsets."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError(
            "validation_fraction must satisfy 0.0 <= validation_fraction < 1.0"
        )

    num_samples = x_train.shape[0]
    if num_samples < 2:
        raise ValueError("Need at least 2 training samples for train/val split")

    raw_val_size = int(round(num_samples * validation_fraction))
    val_size = min(max(raw_val_size, 1), num_samples - 1)

    generator = torch.Generator(device=x_train.device)
    generator.manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)

    val_idx = permutation[:val_size]
    train_idx = permutation[val_size:]
    return (
        x_train[train_idx],
        y_train[train_idx],
        x_train[val_idx],
        y_train[val_idx],
    )


def train_model(
    model: nn.Module,
    data: dict,
    max_epochs: int = 3000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    convergence_threshold: float = 0.99,
    patience: int = 200,
    seed: int = 42,
    validation_fraction: float = 0.2,
    log_interval: int = 500,
) -> dict:
    """Train a model and record metrics.

    Args:
        model: The neural network to train.
        data: Dict from task generators with x_train, y_train, x_test, y_test.
        max_epochs: Maximum training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization.
        convergence_threshold: Accuracy/R2 threshold to consider converged.
        patience: Stop if no improvement for this many epochs.
        seed: Random seed.
        validation_fraction: Fraction of training data used for validation.
        log_interval: Print progress every N epochs.

    Returns:
        Dict with training metrics and sampled curves.
    """
    torch.manual_seed(seed)

    task_type = data["task_type"]
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

    x_train_opt, y_train_opt, x_val, y_val = _split_train_validation(
        x_train,
        y_train,
        validation_fraction=validation_fraction,
        seed=seed,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs, eta_min=lr * 0.01
    )

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Tracking
    train_losses = []
    val_metrics = []
    best_val_metric = -float("inf")
    best_epoch = 0
    convergence_epoch = None
    best_state = None

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # --- Training step ---
        model.train()
        optimizer.zero_grad()
        output = model(x_train_opt)

        if task_type == "regression":
            output = output.squeeze(-1)

        loss = criterion(output, y_train_opt)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        # --- Evaluation step ---
        model.eval()
        with torch.no_grad():
            train_output = model(x_train_opt)
            val_output = model(x_val)
            train_metric, metric_name = _compute_metric(
                train_output, y_train_opt, task_type
            )
            val_metric, _ = _compute_metric(val_output, y_val, task_type)

            val_metrics.append(val_metric)

            if val_metric > best_val_metric:
                best_val_metric = val_metric
                best_epoch = epoch
                best_state = {
                    key: value.detach().clone()
                    for key, value in model.state_dict().items()
                }

            # Convergence is based on validation metric.
            if (
                convergence_epoch is None
                and val_metric >= convergence_threshold
            ):
                convergence_epoch = epoch

        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}: loss={train_loss:.4f}  "
                f"train_{metric_name}={train_metric:.4f}  "
                f"val_{metric_name}={val_metric:.4f}"
            )

        # Early stopping: no improvement for `patience` epochs
        if epoch - best_epoch >= patience:
            print(f"  Early stop at epoch {epoch} (no improvement for "
                  f"{patience} epochs)")
            break

    training_time = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        final_train_output = model(x_train_opt)
        final_val_output = model(x_val)
        final_test_output = model(x_test)
        final_train_metric, _ = _compute_metric(
            final_train_output, y_train_opt, task_type
        )
        final_val_metric, _ = _compute_metric(final_val_output, y_val, task_type)
        final_test_metric, _ = _compute_metric(
            final_test_output, y_test, task_type
        )

    return {
        "final_train_metric": final_train_metric,
        "final_val_metric": final_val_metric,
        "final_test_metric": final_test_metric,
        # Keep this key for downstream compatibility; it now reflects
        # test performance of the checkpoint selected on validation.
        "best_test_metric": final_test_metric,
        "best_val_metric": best_val_metric,
        "best_epoch": best_epoch,
        "convergence_epoch": convergence_epoch,
        "total_epochs": epoch,
        "final_loss": train_loss,
        "metric_name": metric_name,
        "training_time_sec": round(training_time, 2),
        "epoch_losses": train_losses,
        "epoch_val_metrics": val_metrics,
        # Backward-compatible alias expected by older report code.
        "epoch_test_metrics": val_metrics,
        "val_split_fraction": validation_fraction,
        "train_size": int(x_train_opt.shape[0]),
        "val_size": int(x_val.shape[0]),
        "test_size": int(x_test.shape[0]),
    }
