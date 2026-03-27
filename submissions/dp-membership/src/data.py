"""Synthetic data generation for membership inference experiments.

Creates Gaussian cluster classification datasets with controlled splits
for member/non-member distinction needed by shadow model attacks.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate_gaussian_clusters(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    cluster_std: float = 1.5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data from Gaussian clusters.

    Args:
        n_samples: Total number of samples.
        n_features: Dimensionality of each sample.
        n_classes: Number of classes (cluster centers).
        cluster_std: Standard deviation of each cluster.
        seed: Random seed for reproducibility.

    Returns:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label vector of shape (n_samples,).
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    # Generate class centers (spread controls overlap; smaller = harder task)
    centers = rng.randn(n_classes, n_features) * 2.0

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
    return X[perm], y[perm]


def make_member_nonmember_split(
    X: np.ndarray,
    y: np.ndarray,
    member_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """Split data into member (training) and non-member (holdout) sets.

    Args:
        X: Feature matrix.
        y: Label vector.
        member_ratio: Fraction of data used as members (training set).
        seed: Random seed.

    Returns:
        member_dataset: TensorDataset for training (members).
        nonmember_dataset: TensorDataset for holdout (non-members).
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    perm = rng.permutation(n)
    n_member = int(n * member_ratio)

    member_idx = perm[:n_member]
    nonmember_idx = perm[n_member:]

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    member_ds = TensorDataset(X_tensor[member_idx], y_tensor[member_idx])
    nonmember_ds = TensorDataset(X_tensor[nonmember_idx], y_tensor[nonmember_idx])

    return member_ds, nonmember_ds
