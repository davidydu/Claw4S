"""MLP model definitions and training utilities.

Defines 2-layer MLPs of varying hidden widths for the membership
inference scaling experiment. All training is CPU-only with
deterministic seeding.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, List


# Model sizes to evaluate
HIDDEN_WIDTHS = [16, 32, 64, 128, 256]
TRAIN_EPOCHS = 50
LEARNING_RATE = 0.01
BATCH_SIZE = 64
SEED = 42


class TwoLayerMLP(nn.Module):
    """Two-layer MLP for classification.

    Architecture: input -> Linear -> ReLU -> Linear -> output
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def train_model(
    model: TwoLayerMLP,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = TRAIN_EPOCHS,
    lr: float = LEARNING_RATE,
    batch_size: int = BATCH_SIZE,
    seed: int = SEED,
) -> List[float]:
    """Train an MLP model on the given data.

    Args:
        model: The MLP model to train.
        X_train: Training features as numpy array.
        y_train: Training labels as numpy array.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Mini-batch size.
        seed: Random seed for reproducibility.

    Returns:
        List of per-epoch training losses.
    """
    set_seed(seed)

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    n = len(X_tensor)

    for epoch in range(epochs):
        # Shuffle data each epoch
        perm = torch.randperm(n)
        X_shuffled = X_tensor[perm]
        y_shuffled = y_tensor[perm]

        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n, batch_size):
            X_batch = X_shuffled[i : i + batch_size]
            y_batch = y_shuffled[i : i + batch_size]

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        losses.append(epoch_loss / n_batches)

    return losses


def get_predictions(
    model: TwoLayerMLP,
    X: np.ndarray,
) -> np.ndarray:
    """Get softmax prediction probabilities from a trained model.

    Probabilities are clipped to [1e-7, 1-1e-7] and renormalized to avoid
    numerical issues in downstream logistic regression attack classifiers.

    Args:
        model: Trained MLP model.
        X: Feature array.

    Returns:
        Softmax probability array of shape (n_samples, n_classes).
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=1).numpy()

    # Clip extreme probabilities to prevent overflow in attack classifier
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)
    # Renormalize rows to sum to 1
    probs = probs / probs.sum(axis=1, keepdims=True)
    return probs


def compute_accuracy(
    model: TwoLayerMLP,
    X: np.ndarray,
    y: np.ndarray,
) -> float:
    """Compute classification accuracy.

    Args:
        model: Trained MLP model.
        X: Feature array.
        y: Label array.

    Returns:
        Accuracy as a float in [0, 1].
    """
    probs = get_predictions(model, X)
    preds = np.argmax(probs, axis=1)
    return float(np.mean(preds == y))


def create_and_train_model(
    hidden_width: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    input_dim: int = 10,
    output_dim: int = 5,
    seed: int = SEED,
    epochs: int = TRAIN_EPOCHS,
) -> Tuple[TwoLayerMLP, List[float]]:
    """Create and train a 2-layer MLP.

    Args:
        hidden_width: Number of hidden units.
        X_train: Training features.
        y_train: Training labels.
        input_dim: Input feature dimension.
        output_dim: Number of classes.
        seed: Random seed.
        epochs: Training epochs.

    Returns:
        Trained model and list of training losses.
    """
    set_seed(seed)
    model = TwoLayerMLP(input_dim, hidden_width, output_dim)
    losses = train_model(model, X_train, y_train, epochs=epochs, seed=seed)
    return model, losses
