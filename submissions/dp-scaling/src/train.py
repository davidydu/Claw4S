"""Training routines: standard SGD and DP-SGD (from scratch).

DP-SGD implements per-sample gradient clipping and Gaussian noise addition
without relying on external DP libraries, ensuring full transparency and
reproducibility of the privacy mechanism.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model import MLP, count_parameters


def train_standard(
    model: MLP,
    train_loader: DataLoader,
    lr: float = 0.01,
    epochs: int = 100,
    seed: int = 42,
) -> MLP:
    """Train model with standard SGD (no privacy).

    Args:
        model: MLP model to train.
        train_loader: Training data loader.
        lr: Learning rate.
        epochs: Number of training epochs.
        seed: Random seed for reproducibility.

    Returns:
        Trained model.
    """
    torch.manual_seed(seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

    return model


def _compute_per_sample_gradients(
    model: MLP,
    X_batch: torch.Tensor,
    y_batch: torch.Tensor,
    criterion: nn.Module,
) -> list[list[torch.Tensor]]:
    """Compute per-sample gradients using a loop over individual samples.

    For each sample, computes the gradient of the loss with respect to all
    model parameters. This is the core primitive for DP-SGD: we need
    individual gradients so we can clip each one independently.

    Args:
        model: The model.
        X_batch: Input batch of shape (batch_size, n_features).
        y_batch: Label batch of shape (batch_size,).
        criterion: Loss function (applied per-sample with reduction='none'-style).

    Returns:
        List of length batch_size, where each element is a list of gradient
        tensors (one per model parameter), matching the order of
        model.parameters().
    """
    per_sample_grads = []
    for i in range(len(X_batch)):
        model.zero_grad()
        logit = model(X_batch[i : i + 1])
        loss_i = criterion(logit, y_batch[i : i + 1])
        loss_i.backward()
        grads_i = [p.grad.clone() for p in model.parameters()]
        per_sample_grads.append(grads_i)
    return per_sample_grads


def _clip_and_noise(
    per_sample_grads: list[list[torch.Tensor]],
    max_grad_norm: float,
    noise_multiplier: float,
    generator: torch.Generator,
) -> list[torch.Tensor]:
    """Clip per-sample gradients and add calibrated Gaussian noise.

    Implements the DP-SGD mechanism:
    1. Clip each per-sample gradient to L2 norm <= max_grad_norm.
    2. Sum the clipped gradients.
    3. Add Gaussian noise with std = max_grad_norm * noise_multiplier.
    4. Divide by batch size to get the noisy average gradient.

    Args:
        per_sample_grads: Per-sample gradient lists from
            _compute_per_sample_gradients.
        max_grad_norm: Maximum L2 norm for gradient clipping (C).
        noise_multiplier: Noise multiplier (sigma). Noise std =
            max_grad_norm * noise_multiplier.
        generator: Torch random generator for reproducible noise.

    Returns:
        List of noisy average gradient tensors, one per model parameter.
    """
    batch_size = len(per_sample_grads)
    n_params = len(per_sample_grads[0])

    # Initialize aggregated gradients
    summed = [torch.zeros_like(per_sample_grads[0][j]) for j in range(n_params)]

    for i in range(batch_size):
        # Compute L2 norm of this sample's full gradient
        grad_norm = torch.sqrt(
            sum(g.norm() ** 2 for g in per_sample_grads[i])
        )
        # Clip factor
        clip_factor = min(1.0, max_grad_norm / (grad_norm.item() + 1e-8))
        for j in range(n_params):
            summed[j] += per_sample_grads[i][j] * clip_factor

    # Add Gaussian noise and average
    noise_std = max_grad_norm * noise_multiplier
    for j in range(n_params):
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=summed[j].shape,
            generator=generator,
        )
        summed[j] = (summed[j] + noise) / batch_size

    return summed


def train_dp_sgd(
    model: MLP,
    train_loader: DataLoader,
    lr: float = 0.01,
    epochs: int = 100,
    max_grad_norm: float = 1.0,
    noise_multiplier: float = 1.0,
    seed: int = 42,
) -> MLP:
    """Train model with DP-SGD (per-sample clipping + Gaussian noise).

    Implements differentially private stochastic gradient descent from
    scratch, without external DP libraries. Each gradient update:
    1. Computes per-sample gradients.
    2. Clips each to L2 norm <= max_grad_norm.
    3. Sums clipped gradients and adds Gaussian noise.
    4. Applies the noisy averaged gradient as the parameter update.

    Args:
        model: MLP model to train.
        train_loader: Training data loader.
        lr: Learning rate.
        epochs: Number of training epochs.
        max_grad_norm: Clipping bound C for per-sample gradients.
        noise_multiplier: Noise multiplier sigma (noise std = C * sigma).
        seed: Random seed for reproducibility.

    Returns:
        Trained model.
    """
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 1000)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            # Step 1: Per-sample gradients
            per_sample_grads = _compute_per_sample_gradients(
                model, X_batch, y_batch, criterion
            )

            # Step 2-3: Clip and add noise
            noisy_grads = _clip_and_noise(
                per_sample_grads, max_grad_norm, noise_multiplier, generator
            )

            # Step 4: Apply noisy gradient
            model.zero_grad()
            for param, noisy_grad in zip(model.parameters(), noisy_grads):
                param.data -= lr * noisy_grad

    return model


def evaluate(model: MLP, test_loader: DataLoader) -> tuple[float, float]:
    """Evaluate model on test set.

    Args:
        model: Trained model.
        test_loader: Test data loader.

    Returns:
        test_loss: Average cross-entropy loss on the test set.
        accuracy: Classification accuracy on the test set.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += len(y_batch)

    return total_loss / total, correct / total
