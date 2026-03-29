"""Training loop with per-layer gradient and weight norm tracking.

Records at every epoch:
  - Per-layer gradient L2 norms (after backward, before optimizer step)
  - Per-layer weight L2 norms
  - Train/test loss
  - Train/test accuracy (classification) or R^2 (regression)
"""

import torch
import torch.nn as nn
import numpy as np
from src.models import TwoLayerMLP


def _accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def _r_squared(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute R-squared for regression."""
    ss_res = ((preds - targets) ** 2).sum().item()
    ss_tot = ((targets - targets.mean()) ** 2).sum().item()
    if ss_tot < 1e-12:
        return 1.0
    return 1.0 - ss_res / ss_tot


def train_and_track(
    dataset: dict,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.1,
    n_epochs: int = 3000,
    seed: int = 42,
    log_every: int = 1,
) -> dict:
    """Train a 2-layer MLP and record training dynamics.

    Args:
        dataset: dict from data.make_*_dataset().
        hidden_dim: hidden layer width.
        lr: learning rate for AdamW.
        weight_decay: weight decay for AdamW.
        n_epochs: number of training epochs.
        seed: random seed for reproducibility.
        log_every: record metrics every N epochs.

    Returns:
        dict with keys:
            'epochs': list of epoch numbers
            'grad_norms': {layer_name: [norm_at_each_logged_epoch]}
            'weight_norms': {layer_name: [norm_at_each_logged_epoch]}
            'train_loss': [float]
            'test_loss': [float]
            'train_metric': [float]  (accuracy or R^2)
            'test_metric': [float]
            'metric_name': str  ('accuracy' or 'r_squared')
            'task_name': str
            'frac': float  (training fraction, inferred)
            'config': dict of hyperparameters
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    x_train = dataset["x_train"]
    y_train = dataset["y_train"]
    x_test = dataset["x_test"]
    y_test = dataset["y_test"]
    input_dim = dataset["input_dim"]
    output_dim = dataset["output_dim"]
    task_name = dataset["task_name"]

    is_classification = (task_name == "modular_addition")
    metric_name = "accuracy" if is_classification else "r_squared"

    model = TwoLayerMLP(input_dim, hidden_dim, output_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if is_classification:
        loss_fn = nn.CrossEntropyLoss()
    else:
        loss_fn = nn.MSELoss()

    layer_names = model.get_layer_names()

    # Storage
    epochs_log: list[int] = []
    grad_norms: dict[str, list[float]] = {name: [] for name in layer_names}
    weight_norms: dict[str, list[float]] = {name: [] for name in layer_names}
    train_losses: list[float] = []
    test_losses: list[float] = []
    train_metrics: list[float] = []
    test_metrics: list[float] = []

    for epoch in range(n_epochs):
        # --- Training step ---
        model.train()
        optimizer.zero_grad()
        train_out = model(x_train)
        train_loss = loss_fn(train_out, y_train)
        train_loss.backward()

        # Record gradient norms BEFORE optimizer step
        if epoch % log_every == 0:
            epochs_log.append(epoch)
            train_losses.append(train_loss.item())

            for name in layer_names:
                layer = getattr(model, name)
                g_norm = 0.0
                w_norm = 0.0
                for p in layer.parameters():
                    if p.grad is not None:
                        g_norm += p.grad.data.norm(2).item() ** 2
                    w_norm += p.data.norm(2).item() ** 2
                grad_norms[name].append(g_norm ** 0.5)
                weight_norms[name].append(w_norm ** 0.5)

            # Train metric
            with torch.no_grad():
                if is_classification:
                    train_metrics.append(_accuracy(train_out, y_train))
                else:
                    train_metrics.append(_r_squared(train_out, y_train))

            # Test metrics
            model.eval()
            with torch.no_grad():
                test_out = model(x_test)
                test_loss = loss_fn(test_out, y_test)
                test_losses.append(test_loss.item())
                if is_classification:
                    test_metrics.append(_accuracy(test_out, y_test))
                else:
                    test_metrics.append(_r_squared(test_out, y_test))

        optimizer.step()

    n_total = len(x_train) + len(x_test)
    frac = len(x_train) / n_total

    return {
        "epochs": epochs_log,
        "grad_norms": grad_norms,
        "weight_norms": weight_norms,
        "train_loss": train_losses,
        "test_loss": test_losses,
        "train_metric": train_metrics,
        "test_metric": test_metrics,
        "metric_name": metric_name,
        "task_name": task_name,
        "frac": frac,
        "config": {
            "hidden_dim": hidden_dim,
            "lr": lr,
            "weight_decay": weight_decay,
            "n_epochs": n_epochs,
            "seed": seed,
            "deterministic_algorithms": True,
            "torch_version": torch.__version__,
        },
    }
