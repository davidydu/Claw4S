"""Training loop with activation sparsity tracking.

Trains a ReLU MLP and records sparsity metrics at regular intervals.
"""

import torch
import torch.nn as nn
from src.models import ReLUMLP
from src.metrics import compute_all_metrics


def train_with_tracking(
    model: ReLUMLP,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    task_type: str,
    n_epochs: int = 3000,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    track_every: int = 50,
    probe_batch_size: int = 512,
    seed: int = 42,
) -> dict:
    """Train a model and track activation sparsity at regular intervals.

    Parameters
    ----------
    model : ReLUMLP
        The model to train.
    x_train, y_train : torch.Tensor
        Training data.
    x_test, y_test : torch.Tensor
        Test data for generalization measurement.
    task_type : str
        Either 'classification' or 'regression'. Determines loss and accuracy metric.
    n_epochs : int
        Total training epochs.
    lr : float
        Learning rate for AdamW.
    weight_decay : float
        Weight decay for AdamW.
    track_every : int
        Record metrics every this many epochs.
    probe_batch_size : int
        Number of samples to use for activation probing.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: 'epochs', 'train_loss', 'test_loss', 'train_acc', 'test_acc',
              'dead_neuron_fraction', 'near_dead_fraction', 'zero_fraction',
              'activation_entropy', 'mean_activation_magnitude'.
        Each value is a list of measurements taken every `track_every` epochs.
    """
    torch.manual_seed(seed)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    if task_type == "classification":
        criterion = nn.CrossEntropyLoss()
    elif task_type == "regression":
        criterion = nn.MSELoss()
    else:
        raise ValueError(f"Unknown task_type: {task_type!r}")

    # Prepare probe batch (subset for computing sparsity metrics)
    n_probe = min(probe_batch_size, x_train.shape[0])
    x_probe = x_train[:n_probe]

    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
        "dead_neuron_fraction": [],
        "near_dead_fraction": [],
        "zero_fraction": [],
        "activation_entropy": [],
        "mean_activation_magnitude": [],
    }

    for epoch in range(n_epochs + 1):
        # Training step (skip at epoch 0 to record initial state)
        if epoch > 0:
            model.train()
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

        # Record metrics at tracking intervals
        if epoch % track_every == 0:
            model.eval()
            with torch.no_grad():
                # Training metrics
                train_out = model(x_train)
                train_loss = criterion(train_out, y_train).item()

                # Test metrics
                test_out = model(x_test)
                test_loss = criterion(test_out, y_test).item()

                # Accuracy
                if task_type == "classification":
                    train_acc = (train_out.argmax(dim=1) == y_train).float().mean().item()
                    test_acc = (test_out.argmax(dim=1) == y_test).float().mean().item()
                else:
                    # For regression, use R^2 as "accuracy"
                    train_var = y_train.var().item()
                    test_var = y_test.var().item()
                    train_acc = max(0.0, 1.0 - train_loss / train_var) if train_var > 0 else 0.0
                    test_acc = max(0.0, 1.0 - test_loss / test_var) if test_var > 0 else 0.0

                # Sparsity metrics via probe batch
                _ = model(x_probe)
                activations = model.get_last_hidden()
                sparsity = compute_all_metrics(activations)

            history["epochs"].append(epoch)
            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["dead_neuron_fraction"].append(sparsity["dead_neuron_fraction"])
            history["near_dead_fraction"].append(sparsity["near_dead_fraction"])
            history["zero_fraction"].append(sparsity["zero_fraction"])
            history["activation_entropy"].append(sparsity["activation_entropy"])
            history["mean_activation_magnitude"].append(sparsity["mean_activation_magnitude"])

    return history
