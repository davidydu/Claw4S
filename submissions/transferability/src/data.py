"""Synthetic Gaussian-cluster classification dataset."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_gaussian_clusters(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
    cluster_std: float = 1.0,
) -> TensorDataset:
    """Generate synthetic Gaussian cluster data for classification.

    Each class is a Gaussian blob with a distinct centroid.
    Centroids are spaced along the unit hypersphere to ensure separability.
    When n_samples is not divisible by n_classes, the remainder is
    distributed deterministically across early classes so output size
    is always exactly n_samples.

    Args:
        n_samples: Total number of samples across all classes.
        n_features: Dimensionality of each sample.
        n_classes: Number of classes.
        seed: Random seed for reproducibility.
        cluster_std: Standard deviation of each cluster.

    Returns:
        TensorDataset with (X, y) where X is float32 and y is int64.
    """
    rng = np.random.RandomState(seed)
    base_samples_per_class = n_samples // n_classes
    remainder = n_samples % n_classes

    # Generate well-separated centroids using orthogonal directions
    # Use QR decomposition for orthogonal centroid directions
    raw = rng.randn(n_features, n_features)
    q, _ = np.linalg.qr(raw)
    centroids = q[:n_classes] * 3.0  # scale for separation

    xs, ys = [], []
    for c in range(n_classes):
        n_c = base_samples_per_class + (1 if c < remainder else 0)
        x_c = rng.randn(n_c, n_features) * cluster_std + centroids[c]
        xs.append(x_c)
        ys.append(np.full(n_c, c, dtype=np.int64))

    X = np.concatenate(xs, axis=0).astype(np.float32)
    y = np.concatenate(ys, axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
