"""Two-layer MLP classifier for data poisoning experiments."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation.

    Architecture: input -> Linear(hidden) -> ReLU -> Linear(n_classes)

    Args:
        n_features: Number of input features.
        hidden_width: Number of hidden units.
        n_classes: Number of output classes.
    """

    def __init__(self, n_features: int, hidden_width: int, n_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_width),
            nn.ReLU(),
            nn.Linear(hidden_width, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_model(
    model: MLP,
    train_dataset: TensorDataset,
    n_epochs: int = 100,
    lr: float = 0.01,
    batch_size: int = 64,
    seed: int = 42,
) -> list[float]:
    """Train the MLP with cross-entropy loss and SGD.

    Args:
        model: The MLP to train.
        train_dataset: Training data (features, labels).
        n_epochs: Number of training epochs.
        lr: Learning rate for SGD.
        batch_size: Mini-batch size.
        seed: Random seed for DataLoader shuffling.

    Returns:
        List of per-epoch training losses.
    """
    torch.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    losses = []
    for _epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        losses.append(epoch_loss / max(n_batches, 1))

    return losses


@torch.no_grad()
def evaluate_accuracy(
    model: MLP,
    X: torch.Tensor,
    y: torch.Tensor,
) -> float:
    """Compute classification accuracy.

    Args:
        model: Trained MLP.
        X: Feature tensor.
        y: Label tensor.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    logits = model(X)
    preds = logits.argmax(dim=1)
    accuracy = (preds == y).float().mean().item()
    model.train()
    return accuracy
