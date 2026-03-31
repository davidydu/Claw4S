"""Synthetic data generation for double descent experiments.

Generates noisy regression data: y = X @ w_true + epsilon,
where X ~ N(0,1) and epsilon ~ N(0, noise_std).
"""

import torch


def generate_regression_data(
    n_train: int = 200,
    n_test: int = 200,
    d: int = 20,
    noise_std: float = 1.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic noisy linear regression data.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        d: Input dimensionality.
        noise_std: Standard deviation of label noise.
        seed: Random seed for reproducibility.

    Returns:
        (X_train, y_train, X_test, y_test) as float32 tensors.
        X shapes: (n, d), y shapes: (n, 1).
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    n_total = n_train + n_test

    # Generate features: X ~ N(0, 1)
    X = torch.randn(n_total, d, generator=gen)

    # True weights
    w_true = torch.randn(d, 1, generator=gen)

    # Clean signal
    y_clean = X @ w_true

    # Add noise
    noise = torch.randn(n_total, 1, generator=gen) * noise_std
    y = y_clean + noise

    # Split into train/test
    X_train = X[:n_train].clone()
    y_train = y[:n_train].clone()
    X_test = X[n_train:].clone()
    y_test = y[n_train:].clone()

    return X_train, y_train, X_test, y_test


def get_data_summary(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
) -> dict:
    """Return a summary of the dataset dimensions and statistics."""
    return {
        "n_train": X_train.shape[0],
        "n_test": X_test.shape[0],
        "d": X_train.shape[1],
        "y_train_mean": float(y_train.mean()),
        "y_train_std": float(y_train.std()),
        "y_test_mean": float(y_test.mean()),
        "y_test_std": float(y_test.std()),
    }
