"""Synthetic data generation for DP scaling law experiments.

Generates Gaussian cluster classification data with reproducible seeding.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def generate_gaussian_clusters(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
    cluster_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data from Gaussian clusters.

    Each class is centered at a random point in feature space, with samples
    drawn from a Gaussian distribution around the center.

    Args:
        n_samples: Total number of samples across all classes.
        n_features: Number of input features.
        n_classes: Number of classes (Gaussian clusters).
        seed: Random seed for reproducibility.
        cluster_std: Standard deviation of each cluster.

    Returns:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label vector of shape (n_samples,) with values in [0, n_classes).
    """
    rng = np.random.RandomState(seed)

    # Generate class centers spread out in feature space
    centers = rng.randn(n_classes, n_features) * 3.0

    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

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
    X = X[perm]
    y = y[perm]

    return X, y


def make_dataloaders(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
    batch_size: int = 64,
    train_fraction: float = 0.8,
) -> tuple[DataLoader, DataLoader, int, int]:
    """Create train/test DataLoaders from synthetic Gaussian clusters.

    Args:
        n_samples: Total number of samples.
        n_features: Number of input features.
        n_classes: Number of classes.
        seed: Random seed.
        batch_size: Batch size for DataLoaders.
        train_fraction: Fraction of data used for training.

    Returns:
        train_loader: DataLoader for training set.
        test_loader: DataLoader for test set.
        n_features: Number of input features (for model construction).
        n_classes: Number of output classes (for model construction).
    """
    X, y = generate_gaussian_clusters(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        seed=seed,
    )

    n_train = int(len(X) * train_fraction)

    X_train = torch.tensor(X[:n_train], dtype=torch.float32)
    y_train = torch.tensor(y[:n_train], dtype=torch.long)
    X_test = torch.tensor(X[n_train:], dtype=torch.float32)
    y_test = torch.tensor(y[n_train:], dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, n_features, n_classes
