"""Synthetic data generation and label poisoning utilities."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate_gaussian_clusters(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
    cluster_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data with Gaussian clusters.

    Each class is a Gaussian blob centered at a random location.

    Args:
        n_samples: Total number of samples across all classes.
        n_features: Dimensionality of feature space.
        n_classes: Number of classes (Gaussian clusters).
        seed: Random seed for reproducibility.
        cluster_std: Standard deviation of each cluster.

    Returns:
        X: Feature array of shape (n_samples, n_features).
        y: Label array of shape (n_samples,) with values in [0, n_classes).
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    # Generate cluster centers with moderate separation
    centers = rng.randn(n_classes, n_features) * 2.0

    X_parts = []
    y_parts = []
    for c in range(n_classes):
        n_c = samples_per_class + (1 if c < remainder else 0)
        X_c = rng.randn(n_c, n_features) * cluster_std + centers[c]
        y_c = np.full(n_c, c, dtype=np.int64)
        X_parts.append(X_c)
        y_parts.append(y_c)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def poison_labels(
    y: np.ndarray,
    poison_fraction: float,
    n_classes: int = 5,
    seed: int = 42,
) -> np.ndarray:
    """Randomly flip a fraction of labels to incorrect classes.

    Args:
        y: Original label array of shape (n_samples,).
        poison_fraction: Fraction of labels to flip, in [0, 1].
        n_classes: Number of classes (used to pick wrong label).
        seed: Random seed for reproducibility.

    Returns:
        y_poisoned: Copy of y with poison_fraction of labels flipped.
    """
    if poison_fraction < 0.0 or poison_fraction > 1.0:
        raise ValueError(f"poison_fraction must be in [0, 1], got {poison_fraction}")

    rng = np.random.RandomState(seed)
    y_poisoned = y.copy()
    n = len(y)
    n_poison = int(round(n * poison_fraction))

    if n_poison == 0:
        return y_poisoned

    poison_idx = rng.choice(n, size=n_poison, replace=False)
    for idx in poison_idx:
        original = y_poisoned[idx]
        # Pick a different label uniformly at random
        wrong_labels = [c for c in range(n_classes) if c != original]
        y_poisoned[idx] = rng.choice(wrong_labels)

    return y_poisoned


def make_datasets(
    X: np.ndarray,
    y_train_labels: np.ndarray,
    y_clean_labels: np.ndarray,
    train_fraction: float = 0.7,
    seed: int = 42,
) -> dict:
    """Split data into train/test TensorDatasets.

    Training set uses (possibly poisoned) y_train_labels.
    Test set always uses clean y_clean_labels.

    Args:
        X: Feature array of shape (n_samples, n_features).
        y_train_labels: Labels for training (may be poisoned).
        y_clean_labels: Clean labels (used for test set and train-clean eval).
        train_fraction: Fraction of data for training.
        seed: Random seed for the split.

    Returns:
        Dictionary with keys: train_dataset, test_dataset,
        train_clean_labels (clean labels for training indices).
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_train = int(round(n * train_fraction))

    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y_train_labels[train_idx], dtype=torch.long)
    y_train_clean = torch.tensor(y_clean_labels[train_idx], dtype=torch.long)

    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y_clean_labels[test_idx], dtype=torch.long)

    return {
        "train_dataset": TensorDataset(X_train, y_train),
        "test_dataset": TensorDataset(X_test, y_test),
        "train_clean_labels": y_train_clean,
        "X_train": X_train,
        "X_test": X_test,
        "y_test": y_test,
    }
