"""Synthetic data generation for membership inference experiments.

Generates Gaussian cluster classification data with deterministic seeding
for reproducibility. Data is split into target-train, target-test,
and shadow-train/test partitions.
"""

import numpy as np
from typing import Tuple


# Fixed configuration
N_SAMPLES = 500
N_FEATURES = 10
N_CLASSES = 5
SEED = 42


def generate_gaussian_clusters(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    seed: int = SEED,
    center_spread: float = 1.0,
    noise_scale: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic classification data with Gaussian clusters.

    Each class center is drawn from N(0, center_spread), and samples
    are drawn from N(center, noise_scale) for each class. The default
    parameters create overlapping clusters that are difficult enough
    to classify that small models underfit while large models overfit.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features per sample.
        n_classes: Number of classes.
        seed: Random seed for reproducibility.
        center_spread: Standard deviation for class center placement.
        noise_scale: Standard deviation of within-class noise.

    Returns:
        X: Feature array of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes

    centers = rng.randn(n_classes, n_features) * center_spread

    X_parts = []
    y_parts = []
    for cls_idx in range(n_classes):
        X_cls = rng.randn(samples_per_class, n_features) * noise_scale + centers[cls_idx]
        y_cls = np.full(samples_per_class, cls_idx, dtype=np.int64)
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle deterministically
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    train_fraction: float = 0.5,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train/test sets.

    Args:
        X: Feature array.
        y: Label array.
        train_fraction: Fraction of data to use for training.
        seed: Random seed for reproducibility.

    Returns:
        X_train, y_train, X_test, y_test
    """
    rng = np.random.RandomState(seed)
    n = len(X)
    perm = rng.permutation(n)
    split_idx = int(n * train_fraction)

    train_idx = perm[:split_idx]
    test_idx = perm[split_idx:]

    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


def generate_shadow_data(
    n_samples: int = N_SAMPLES,
    n_features: int = N_FEATURES,
    n_classes: int = N_CLASSES,
    shadow_idx: int = 0,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate independent shadow dataset for shadow model training.

    Uses a different seed derived from the base seed and shadow index
    to ensure independence from target data.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features per sample.
        n_classes: Number of classes.
        shadow_idx: Index of the shadow model (0, 1, 2, ...).
        seed: Base random seed.

    Returns:
        X: Feature array of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
    """
    shadow_seed = seed + 1000 + shadow_idx * 100
    return generate_gaussian_clusters(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        seed=shadow_seed,
    )
