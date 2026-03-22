"""Synthetic data generation for lottery ticket experiments.

Two tasks:
1. Modular arithmetic: learn (a + b) mod 97 from one-hot inputs
2. Regression: learn a random linear mapping with noise
"""

import torch
import numpy as np


def generate_modular_data(
    mod: int = 97, seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate modular addition dataset: (a + b) mod p.

    Creates all mod^2 pairs, encodes as one-hot(a) || one-hot(b),
    and splits 80/20 into train/test.

    Args:
        mod: The modulus (default 97, a prime for clean structure).
        seed: Random seed for train/test split.

    Returns:
        (X_train, y_train, X_test, y_test) tensors.
    """
    rng = np.random.RandomState(seed)

    # Generate all pairs
    pairs = [(a, b) for a in range(mod) for b in range(mod)]
    n = len(pairs)

    # One-hot encode: input is concat of one-hot(a) and one-hot(b)
    X = torch.zeros(n, 2 * mod)
    y = torch.zeros(n, dtype=torch.long)

    for i, (a, b) in enumerate(pairs):
        X[i, a] = 1.0
        X[i, mod + b] = 1.0
        y[i] = (a + b) % mod

    # Shuffle and split
    perm = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = perm[:split]
    test_idx = perm[split:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def generate_regression_data(
    n_samples: int = 200,
    n_features: int = 20,
    noise_std: float = 0.1,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate random-features regression dataset.

    y = X @ w_true + noise, where w_true is drawn from N(0, 1).
    Split 80/20 into train/test.

    Args:
        n_samples: Total number of samples.
        n_features: Number of input features.
        noise_std: Standard deviation of Gaussian noise.
        seed: Random seed for reproducibility.

    Returns:
        (X_train, y_train, X_test, y_test) tensors.
    """
    rng = np.random.RandomState(seed)

    # True weights
    w_true = rng.randn(n_features).astype(np.float32)

    # Generate data
    X = rng.randn(n_samples, n_features).astype(np.float32)
    noise = noise_std * rng.randn(n_samples).astype(np.float32)
    y = X @ w_true + noise

    X = torch.from_numpy(X)
    y = torch.from_numpy(y)

    # Split
    split = int(0.8 * n_samples)
    return X[:split], y[:split], X[split:], y[split:]
