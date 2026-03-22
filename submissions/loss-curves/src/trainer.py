"""Training loop that records per-epoch loss curves."""

import torch
import torch.nn as nn
from src.models import build_model
from src.tasks import TASK_REGISTRY

SEED = 42


def train_run(
    task_name: str,
    hidden_size: int,
    n_epochs: int = 3000,
    lr: float = 1e-3,
    batch_size: int = 512,
    seed: int = SEED,
) -> dict:
    """Train a model and record the loss at every epoch.

    Returns a dict with keys:
        task, hidden_size, epochs (list[int]), losses (list[float]),
        final_loss (float), n_params (int).
    """
    torch.manual_seed(seed)

    task_config = TASK_REGISTRY[task_name]
    X, y = task_config["make_data"](seed=seed)

    model = build_model(task_config, hidden_size)
    n_params = sum(p.numel() for p in model.parameters())

    if task_config["task_type"] == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    n_samples = X.size(0)
    epoch_list = []
    loss_list = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        # Shuffle data each epoch
        perm = torch.randperm(n_samples)
        X_shuf = X[perm]
        y_shuf = y[perm]

        epoch_loss_sum = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            X_batch = X_shuf[start:end]
            y_batch = y_shuf[start:end]

            optimizer.zero_grad()
            out = model(X_batch)

            if task_config["task_type"] == "regression":
                out = out.squeeze(-1)

            loss = criterion(out, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss_sum += loss.item()
            n_batches += 1

        avg_loss = epoch_loss_sum / n_batches
        epoch_list.append(epoch)
        loss_list.append(avg_loss)

    return {
        "task": task_name,
        "hidden_size": hidden_size,
        "epochs": epoch_list,
        "losses": loss_list,
        "final_loss": loss_list[-1],
        "n_params": n_params,
    }
