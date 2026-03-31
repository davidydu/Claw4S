"""Model training with optional DP-SGD.

Provides a unified training loop that works for both standard SGD
and DP-SGD, used to train target models and shadow models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.dp_sgd import DPConfig, dp_sgd_step
from src.model import MLP


def train_model(
    train_dataset: TensorDataset,
    input_dim: int = 10,
    hidden_dim: int = 64,
    num_classes: int = 5,
    dp_config: DPConfig | None = None,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 0.1,
    seed: int = 42,
) -> tuple[MLP, list[float]]:
    """Train a model with optional DP-SGD.

    Args:
        train_dataset: Training data as a TensorDataset.
        input_dim: Number of input features.
        hidden_dim: Hidden layer width.
        num_classes: Number of classes.
        dp_config: DP-SGD configuration. None means non-private.
        epochs: Number of training epochs.
        batch_size: Batch size.
        lr: Learning rate.
        seed: Random seed for weight initialization.

    Returns:
        model: Trained MLP.
        losses: Per-epoch average loss.
    """
    torch.manual_seed(seed)

    model = MLP(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss(reduction="none")

    if dp_config is None:
        dp_config = DPConfig(noise_multiplier=0.0)

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    epoch_losses = []
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in loader:
            loss = dp_sgd_step(model, optimizer, loss_fn, batch_x, batch_y, dp_config)
            total_loss += loss
            n_batches += 1
        epoch_losses.append(total_loss / max(n_batches, 1))

    return model, epoch_losses


def evaluate_model(
    model: MLP,
    dataset: TensorDataset,
) -> tuple[float, float]:
    """Evaluate model accuracy and average loss on a dataset.

    Args:
        model: Trained model.
        dataset: Evaluation dataset.

    Returns:
        accuracy: Classification accuracy (0 to 1).
        avg_loss: Average cross-entropy loss.
    """
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    correct = 0
    total = 0
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            total_loss += loss.item()
            n_batches += 1
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_y.shape[0]

    accuracy = correct / max(total, 1)
    avg_loss = total_loss / max(n_batches, 1)
    return accuracy, avg_loss
