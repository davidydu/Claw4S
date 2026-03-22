"""Data generation for modular arithmetic and regression tasks."""

import torch
import numpy as np


def generate_modular_addition(
    p: int = 97,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> dict:
    """Generate modular addition dataset: (a + b) mod p.

    Args:
        p: Prime modulus (default 97).
        train_fraction: Fraction of all pairs used for training.
        seed: Random seed for reproducibility.

    Returns:
        Dict with X_train, y_train, X_test, y_test as torch tensors,
        plus metadata (p, n_train, n_test).
    """
    rng = np.random.RandomState(seed)

    # Generate all p^2 pairs
    pairs = []
    labels = []
    for a in range(p):
        for b in range(p):
            pairs.append((a, b))
            labels.append((a + b) % p)

    pairs = np.array(pairs)
    labels = np.array(labels)

    # Shuffle and split
    indices = rng.permutation(len(pairs))
    n_train = int(len(pairs) * train_fraction)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # One-hot encode inputs: concatenate one-hot(a) and one-hot(b)
    def one_hot_encode(pair_array: np.ndarray) -> torch.Tensor:
        n = len(pair_array)
        encoded = np.zeros((n, 2 * p), dtype=np.float32)
        for i, (a, b) in enumerate(pair_array):
            encoded[i, a] = 1.0
            encoded[i, p + b] = 1.0
        return torch.from_numpy(encoded)

    X_train = one_hot_encode(pairs[train_idx])
    y_train = torch.from_numpy(labels[train_idx].astype(np.int64))
    X_test = one_hot_encode(pairs[test_idx])
    y_test = torch.from_numpy(labels[test_idx].astype(np.int64))

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "p": p,
        "input_dim": 2 * p,
        "output_dim": p,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "task": "classification",
    }


def generate_regression(
    n_samples: int = 1000,
    seed: int = 42,
    train_fraction: float = 0.7,
) -> dict:
    """Generate polynomial regression dataset.

    Target: f(x) = 0.5*x^3 - 0.3*x^2 + 0.7*x - 0.1.

    Args:
        n_samples: Total number of samples.
        seed: Random seed for reproducibility.
        train_fraction: Fraction used for training.

    Returns:
        Dict with X_train, y_train, X_test, y_test as torch tensors,
        plus metadata.
    """
    rng = np.random.RandomState(seed)

    x = rng.uniform(-1.0, 1.0, size=n_samples).astype(np.float32)
    y = 0.5 * x**3 - 0.3 * x**2 + 0.7 * x - 0.1

    # Features: [x, x^2, x^3]
    X = np.stack([x, x**2, x**3], axis=1).astype(np.float32)

    # Shuffle and split
    indices = rng.permutation(n_samples)
    n_train = int(n_samples * train_fraction)

    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx]).unsqueeze(1)
    X_test = torch.from_numpy(X[test_idx])
    y_test = torch.from_numpy(y[test_idx]).unsqueeze(1)

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "input_dim": 3,
        "output_dim": 1,
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "task": "regression",
    }
