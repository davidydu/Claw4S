"""Synthetic data generation for feature attribution experiments."""

import numpy as np
from typing import Tuple


def make_gaussian_clusters(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian cluster data for classification.

    Each class centre is drawn from N(0, 3) and samples from N(centre, 1).
    Returns arrays ready for train/test splitting.

    Args:
        n_samples: Total number of samples.
        n_features: Dimensionality of input features.
        n_classes: Number of distinct classes.
        seed: Random seed for reproducibility.

    Returns:
        X: (n_samples, n_features) float32 array.
        y: (n_samples,) int64 array of class labels 0..n_classes-1.
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    X_parts = []
    y_parts = []
    for cls in range(n_classes):
        count = samples_per_class + (1 if cls < remainder else 0)
        centre = rng.randn(n_features) * 3.0
        X_cls = rng.randn(count, n_features) + centre
        X_parts.append(X_cls)
        y_parts.append(np.full(count, cls, dtype=np.int64))

    X = np.concatenate(X_parts, axis=0).astype(np.float32)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle deterministically
    perm = rng.permutation(len(X))
    return X[perm], y[perm]
