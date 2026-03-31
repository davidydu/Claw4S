"""Training loop for grokking experiments.

Implements full-batch training with AdamW optimizer and periodic metric
logging. Supports early stopping when both train and test accuracy
reach high levels.
"""

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from src.model import GrokkingMLP


@dataclass
class TrainResult:
    """Container for training run results."""

    # Metric histories (logged every log_interval epochs)
    train_accs: list[float] = field(default_factory=list)
    test_accs: list[float] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    test_losses: list[float] = field(default_factory=list)
    logged_epochs: list[int] = field(default_factory=list)

    # Milestone epochs (None if threshold not reached)
    epoch_train_95: int | None = None
    epoch_test_95: int | None = None

    # Final metrics
    final_train_acc: float = 0.0
    final_test_acc: float = 0.0
    total_epochs: int = 0


@dataclass
class TrainConfig:
    """Configuration for a training run."""

    lr: float = 1e-3
    weight_decay: float = 0.1
    max_epochs: int = 2500
    log_interval: int = 100
    early_stop_acc: float = 0.99
    early_stop_patience: int = 2  # consecutive checks above threshold
    betas: tuple[float, float] = (0.9, 0.98)
    seed: int = 42


def _compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute classification accuracy."""
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


def train_model(
    model: GrokkingMLP,
    train_data: dict,
    test_data: dict,
    config: TrainConfig | None = None,
) -> TrainResult:
    """Train a model with full-batch gradient descent.

    Args:
        model: The GrokkingMLP to train.
        train_data: Dict with "inputs" and "labels" tensors.
        test_data: Dict with "inputs" and "labels" tensors.
        config: Training configuration. Uses defaults if None.

    Returns:
        TrainResult with metric histories and milestone epochs.
    """
    if config is None:
        config = TrainConfig()

    device = torch.device("cpu")
    model = model.to(device)

    train_inputs = train_data["inputs"].to(device)
    train_labels = train_data["labels"].to(device)
    test_inputs = test_data["inputs"].to(device)
    test_labels = test_data["labels"].to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
        betas=config.betas,
    )
    criterion = nn.CrossEntropyLoss()

    result = TrainResult()
    consecutive_high = 0

    for epoch in range(1, config.max_epochs + 1):
        # Training step
        model.train()
        optimizer.zero_grad()
        logits = model(train_inputs)
        loss = criterion(logits, train_labels)
        loss.backward()
        optimizer.step()

        # Log metrics at intervals and final epoch
        if epoch % config.log_interval == 0 or epoch == config.max_epochs:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_inputs)
                train_loss = criterion(train_logits, train_labels).item()
                train_acc = _compute_accuracy(train_logits, train_labels)

                test_logits = model(test_inputs)
                test_loss = criterion(test_logits, test_labels).item()
                test_acc = _compute_accuracy(test_logits, test_labels)

            result.train_accs.append(train_acc)
            result.test_accs.append(test_acc)
            result.train_losses.append(train_loss)
            result.test_losses.append(test_loss)
            result.logged_epochs.append(epoch)

            # Track milestone epochs (first time reaching 95%)
            if result.epoch_train_95 is None and train_acc >= 0.95:
                result.epoch_train_95 = epoch
            if result.epoch_test_95 is None and test_acc >= 0.95:
                result.epoch_test_95 = epoch

            # Early stopping check
            if train_acc >= config.early_stop_acc and test_acc >= config.early_stop_acc:
                consecutive_high += 1
                if consecutive_high >= config.early_stop_patience:
                    result.total_epochs = epoch
                    result.final_train_acc = train_acc
                    result.final_test_acc = test_acc
                    break
            else:
                consecutive_high = 0

    else:
        # Loop completed without early stopping
        result.total_epochs = config.max_epochs
        if result.train_accs:
            result.final_train_acc = result.train_accs[-1]
            result.final_test_acc = result.test_accs[-1]

    return result
