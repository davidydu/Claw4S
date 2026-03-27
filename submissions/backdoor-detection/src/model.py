"""Two-layer MLP for classification with activation extraction."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation.

    Architecture: input -> Linear(hidden) -> ReLU -> Linear(n_classes)
    The penultimate layer (after ReLU) is used for spectral analysis.

    Args:
        input_dim: Number of input features.
        hidden_dim: Number of hidden units in penultimate layer.
        n_classes: Number of output classes.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning class logits."""
        h = self.relu(self.fc1(x))
        return self.fc2(h)

    def get_penultimate(self, x: torch.Tensor) -> torch.Tensor:
        """Extract penultimate-layer activations (after ReLU).

        Args:
            x: Input tensor of shape (batch_size, input_dim).

        Returns:
            Activation tensor of shape (batch_size, hidden_dim).
        """
        with torch.no_grad():
            return self.relu(self.fc1(x))


def train_model(
    dataset: TensorDataset,
    input_dim: int,
    hidden_dim: int = 128,
    n_classes: int = 5,
    epochs: int = 50,
    lr: float = 0.01,
    batch_size: int = 64,
    seed: int = 42,
) -> MLP:
    """Train a two-layer MLP on the given dataset.

    Uses cross-entropy loss and Adam optimizer. Sets torch manual seed
    for reproducibility.

    Args:
        dataset: TensorDataset with features and labels.
        input_dim: Number of input features.
        hidden_dim: Hidden layer size.
        n_classes: Number of output classes.
        epochs: Number of training epochs.
        lr: Learning rate for Adam.
        batch_size: Mini-batch size.
        seed: Random seed for weight initialization.

    Returns:
        Trained MLP model in eval mode.
    """
    torch.manual_seed(seed)
    model = MLP(input_dim, hidden_dim, n_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for X_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    model.eval()
    return model


def extract_activations(
    model: MLP,
    dataset: TensorDataset,
    batch_size: int = 256,
) -> torch.Tensor:
    """Extract penultimate-layer activations for all samples.

    Args:
        model: Trained MLP model.
        dataset: TensorDataset with features.
        batch_size: Batch size for extraction.

    Returns:
        Activation tensor of shape (n_samples, hidden_dim).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    activations = []
    model.eval()
    with torch.no_grad():
        for X_batch, _ in loader:
            acts = model.get_penultimate(X_batch)
            activations.append(acts)
    return torch.cat(activations, dim=0)
