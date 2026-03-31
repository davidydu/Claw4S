"""Task data generation for Benford's Law experiments.

Generates training/test data for two tasks:
1. Modular arithmetic: predict (a + b) mod p
2. Sine regression: predict sin(x)
"""

import numpy as np
import torch


def generate_modular_data(p=97, seed=42, train_frac=0.7):
    """Generate modular arithmetic dataset: predict (a + b) mod p.

    Args:
        p: Prime modulus.
        seed: Random seed for reproducibility.
        train_frac: Fraction of data for training.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as torch tensors.
        X shape: (N, 2), float32, values in [0, 1].
        y shape: (N,), int64, class labels in [0, p).
    """
    rng = np.random.RandomState(seed)

    # Generate all pairs (a, b) with a, b in [0, p)
    all_pairs = []
    all_labels = []
    for a in range(p):
        for b in range(p):
            all_pairs.append([a, b])
            all_labels.append((a + b) % p)

    all_pairs = np.array(all_pairs, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int64)

    # Normalize inputs to [0, 1]
    all_pairs = all_pairs / (p - 1)

    # Shuffle and split
    n = len(all_pairs)
    indices = rng.permutation(n)
    split = int(n * train_frac)

    X_train = torch.from_numpy(all_pairs[indices[:split]])
    y_train = torch.from_numpy(all_labels[indices[:split]])
    X_test = torch.from_numpy(all_pairs[indices[split:]])
    y_test = torch.from_numpy(all_labels[indices[split:]])

    return X_train, y_train, X_test, y_test


def generate_sine_data(n=1000, seed=42, train_frac=0.7):
    """Generate sine regression dataset: predict sin(x).

    Args:
        n: Number of data points.
        seed: Random seed for reproducibility.
        train_frac: Fraction of data for training.

    Returns:
        Tuple of (X_train, y_train, X_test, y_test) as torch tensors.
        X shape: (N, 1), float32.
        y shape: (N, 1), float32.
    """
    rng = np.random.RandomState(seed)

    x = rng.uniform(0, 2 * np.pi, size=(n, 1)).astype(np.float32)
    y = np.sin(x).astype(np.float32)

    # Shuffle and split
    indices = rng.permutation(n)
    split = int(n * train_frac)

    X_train = torch.from_numpy(x[indices[:split]])
    y_train = torch.from_numpy(y[indices[:split]])
    X_test = torch.from_numpy(x[indices[split:]])
    y_test = torch.from_numpy(y[indices[split:]])

    return X_train, y_train, X_test, y_test
