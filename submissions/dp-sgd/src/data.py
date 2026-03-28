"""Synthetic data generation for DP-SGD experiments.

Generates Gaussian cluster classification data with controlled
separability for privacy-utility tradeoff analysis.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def generate_gaussian_clusters(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    cluster_std: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian cluster data for classification.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features per sample.
        n_classes: Number of classes (Gaussian clusters).
        cluster_std: Standard deviation of each cluster.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, n_features)
        and y has shape (n_samples,) with integer class labels.
    """
    rng = np.random.RandomState(seed)

    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    # Generate well-separated cluster centers
    centers = rng.randn(n_classes, n_features) * 3.0

    X_parts = []
    y_parts = []
    for c in range(n_classes):
        n_c = samples_per_class + (1 if c < remainder else 0)
        X_c = rng.randn(n_c, n_features) * cluster_std + centers[c]
        X_parts.append(X_c)
        y_parts.append(np.full(n_c, c, dtype=np.int64))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    X = X[perm]
    y = y[perm]

    return X, y


def make_dataloaders(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    cluster_std: float = 1.5,
    seed: int = 42,
    test_fraction: float = 0.2,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Create train and test DataLoaders from synthetic data.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features.
        n_classes: Number of classes.
        cluster_std: Cluster standard deviation.
        seed: Random seed.
        test_fraction: Fraction of data for test set.
        batch_size: Batch size for DataLoaders.

    Returns:
        (train_loader, test_loader, n_train, n_test)
    """
    X, y = generate_gaussian_clusters(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        cluster_std=cluster_std,
        seed=seed,
    )

    # Train/test split
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test

    X_train_np = X[:n_train]
    X_test_np = X[n_train:]

    # Fit normalization on the training split only to avoid test leakage.
    mean = X_train_np.mean(axis=0)
    std = X_train_np.std(axis=0) + 1e-8
    X_train_np = (X_train_np - mean) / std
    X_test_np = (X_test_np - mean) / std

    X_train = torch.tensor(X_train_np, dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.long)
    X_test = torch.tensor(X_test_np, dtype=torch.float32)
    y_test = torch.tensor(y[n_train:], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=len(test_ds), shuffle=False)

    return train_loader, test_loader, n_train, n_test
