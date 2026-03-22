"""Training loop for depth-width experiments."""

import time
import torch
import torch.nn as nn


def train_model(
    model: nn.Module,
    data: dict,
    max_epochs: int = 3000,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    convergence_threshold: float = 0.99,
    patience: int = 200,
    seed: int = 42,
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
        log_interval: Print progress every N epochs.

    Returns:
        Dict with training metrics: final_train_metric, final_test_metric,
        convergence_epoch, epoch_losses, best_test_metric, training_time_sec.
    """
    torch.manual_seed(seed)

    task_type = data["task_type"]
    x_train = data["x_train"]
    y_train = data["y_train"]
    x_test = data["x_test"]
    y_test = data["y_test"]

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
    test_metrics = []
    best_test_metric = -float("inf")
    best_epoch = 0
    convergence_epoch = None

    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        # --- Training step ---
        model.train()
        optimizer.zero_grad()
        output = model(x_train)

        if task_type == "regression":
            output = output.squeeze(-1)

        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        train_loss = loss.item()
        train_losses.append(train_loss)

        # --- Evaluation step ---
        model.eval()
        with torch.no_grad():
            test_output = model(x_test)
            train_output = model(x_train)

            if task_type == "classification":
                test_preds = test_output.argmax(dim=1)
                test_metric = (test_preds == y_test).float().mean().item()
                train_preds = train_output.argmax(dim=1)
                train_metric = (train_preds == y_train).float().mean().item()
                metric_name = "accuracy"
            else:
                test_output = test_output.squeeze(-1)
                train_output = train_output.squeeze(-1)
                # R-squared
                ss_res_test = ((y_test - test_output) ** 2).sum().item()
                ss_tot_test = ((y_test - y_test.mean()) ** 2).sum().item()
                test_metric = 1.0 - ss_res_test / max(ss_tot_test, 1e-8)
                ss_res_train = ((y_train - train_output) ** 2).sum().item()
                ss_tot_train = (
                    (y_train - y_train.mean()) ** 2
                ).sum().item()
                train_metric = 1.0 - ss_res_train / max(ss_tot_train, 1e-8)
                metric_name = "r_squared"

            test_metrics.append(test_metric)

            if test_metric > best_test_metric:
                best_test_metric = test_metric
                best_epoch = epoch

            # Check convergence (first time test metric crosses threshold)
            if (
                convergence_epoch is None
                and test_metric >= convergence_threshold
            ):
                convergence_epoch = epoch

        # Logging
        if epoch % log_interval == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:4d}: loss={train_loss:.4f}  "
                f"train_{metric_name}={train_metric:.4f}  "
                f"test_{metric_name}={test_metric:.4f}"
            )

        # Early stopping: no improvement for `patience` epochs
        if epoch - best_epoch >= patience:
            print(f"  Early stop at epoch {epoch} (no improvement for "
                  f"{patience} epochs)")
            break

    training_time = time.time() - start_time

    return {
        "final_train_metric": train_metric,
        "final_test_metric": test_metric,
        "best_test_metric": best_test_metric,
        "best_epoch": best_epoch,
        "convergence_epoch": convergence_epoch,
        "total_epochs": epoch,
        "final_loss": train_loss,
        "metric_name": metric_name,
        "training_time_sec": round(training_time, 2),
        "epoch_losses": train_losses,
        "epoch_test_metrics": test_metrics,
    }
