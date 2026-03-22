"""Dataset generation for gradient norm experiments.

Two tasks:
  1. Modular addition (mod 97) -- grokking-prone classification
  2. Polynomial regression -- smooth learning curve
"""

import torch
import numpy as np


MODULUS = 97  # prime, standard for grokking studies


def make_modular_addition_dataset(
    modulus: int = MODULUS,
    frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """Create modular addition dataset: (a, b) -> (a + b) mod p.

    Args:
        modulus: prime modulus for the addition group.
        frac: fraction of all pairs used for training.
        seed: random seed for reproducibility.

    Returns:
        dict with keys 'x_train', 'y_train', 'x_test', 'y_test',
        'input_dim', 'output_dim', 'task_name'.
    """
    rng = np.random.RandomState(seed)

    # All possible (a, b) pairs
    pairs = []
    labels = []
    for a in range(modulus):
        for b in range(modulus):
            pairs.append((a, b))
            labels.append((a + b) % modulus)

    pairs = np.array(pairs)
    labels = np.array(labels)

    n_total = len(pairs)
    n_train = int(n_total * frac)

    indices = rng.permutation(n_total)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    # One-hot encode inputs: concatenate one-hot(a) and one-hot(b)
    def to_onehot(pair_array: np.ndarray) -> torch.Tensor:
        n = len(pair_array)
        x = np.zeros((n, 2 * modulus), dtype=np.float32)
        for i, (a, b) in enumerate(pair_array):
            x[i, a] = 1.0
            x[i, modulus + b] = 1.0
        return torch.from_numpy(x)

    return {
        "x_train": to_onehot(pairs[train_idx]),
        "y_train": torch.from_numpy(labels[train_idx]).long(),
        "x_test": to_onehot(pairs[test_idx]),
        "y_test": torch.from_numpy(labels[test_idx]).long(),
        "input_dim": 2 * modulus,
        "output_dim": modulus,
        "task_name": "modular_addition",
    }


def make_regression_dataset(
    n_samples: int = 2000,
    frac: float = 0.7,
    seed: int = 42,
) -> dict:
    """Create polynomial regression dataset: y = sin(x) + 0.3*sin(3x).

    A smooth-learning task (no grokking expected).

    Args:
        n_samples: total number of data points.
        frac: fraction used for training.
        seed: random seed for reproducibility.

    Returns:
        dict with keys 'x_train', 'y_train', 'x_test', 'y_test',
        'input_dim', 'output_dim', 'task_name'.
    """
    rng = np.random.RandomState(seed)

    x = rng.uniform(-3.0, 3.0, size=(n_samples, 1)).astype(np.float32)
    y = (np.sin(x) + 0.3 * np.sin(3 * x)).astype(np.float32)

    n_train = int(n_samples * frac)
    indices = rng.permutation(n_samples)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    return {
        "x_train": torch.from_numpy(x[train_idx]),
        "y_train": torch.from_numpy(y[train_idx]),
        "x_test": torch.from_numpy(x[test_idx]),
        "y_test": torch.from_numpy(y[test_idx]),
        "input_dim": 1,
        "output_dim": 1,
        "task_name": "regression",
    }
