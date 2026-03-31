"""Synthetic data generation for calibration experiments.

Generates Gaussian-cluster classification data with controllable distribution
shift. Shift is applied per-class with random directions to break decision
boundaries (not just uniform translation, which preserves separability).
"""

import numpy as np
from typing import Tuple

# Fixed parameters for reproducibility
N_FEATURES = 10
N_CLASSES = 5
N_SAMPLES_TRAIN = 500
N_SAMPLES_TEST = 200
CLUSTER_STD = 1.5  # Higher std for more overlap between clusters
CENTER_SCALE = 2.0  # Closer centers = harder problem


def generate_cluster_centers(n_classes: int, n_features: int,
                             rng: np.random.Generator) -> np.ndarray:
    """Generate cluster centers with moderate separation.

    Centers are spaced to create a non-trivial classification problem:
    close enough that shift causes real misclassification, far enough
    that in-distribution accuracy is reasonable (80-95%).

    Args:
        n_classes: Number of classes/clusters.
        n_features: Dimensionality of feature space.
        rng: NumPy random generator for reproducibility.

    Returns:
        Array of shape (n_classes, n_features) with cluster centers.
    """
    centers = rng.standard_normal((n_classes, n_features)) * CENTER_SCALE
    return centers


def _get_shift_directions(n_classes: int, n_features: int,
                          seed: int) -> np.ndarray:
    """Generate per-class random shift directions (deterministic).

    Each class gets a different random unit direction, so shift breaks
    decision boundaries rather than preserving relative positions.

    Args:
        n_classes: Number of classes.
        n_features: Feature dimensionality.
        seed: Seed for reproducibility.

    Returns:
        Array of shape (n_classes, n_features) with unit shift directions.
    """
    rng = np.random.default_rng(seed + 99999)  # Separate from data rng
    directions = rng.standard_normal((n_classes, n_features))
    # Normalize to unit vectors
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    return directions / norms


def generate_data(centers: np.ndarray, n_samples: int,
                  rng: np.random.Generator,
                  shift_magnitude: float = 0.0,
                  shift_directions: np.ndarray | None = None
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate samples from Gaussian clusters with optional per-class shift.

    Args:
        centers: Cluster centers, shape (n_classes, n_features).
        n_samples: Total number of samples (split evenly across classes).
        rng: NumPy random generator.
        shift_magnitude: How far to translate each cluster mean along its
            shift direction.
        shift_directions: Per-class shift directions, shape (n_classes, n_features).
            If None and shift_magnitude > 0, shifts along the first axis.

    Returns:
        Tuple of (X, y) where X has shape (n_samples, n_features) and
        y has shape (n_samples,) with integer class labels.
    """
    n_classes, n_features = centers.shape
    samples_per_class = n_samples // n_classes

    shifted_centers = centers.copy()
    if shift_magnitude > 0.0:
        if shift_directions is not None:
            shifted_centers = centers + shift_magnitude * shift_directions
        else:
            # Default: shift along first axis (all classes same direction)
            shift_vec = np.zeros(n_features)
            shift_vec[0] = 1.0
            shifted_centers = centers + shift_magnitude * shift_vec

    X_parts = []
    y_parts = []
    for cls_idx in range(n_classes):
        X_cls = rng.standard_normal((samples_per_class, n_features)) * CLUSTER_STD
        X_cls += shifted_centers[cls_idx]
        X_parts.append(X_cls)
        y_parts.append(np.full(samples_per_class, cls_idx, dtype=np.int64))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def make_datasets(seed: int = 42,
                  shift_magnitudes: list[float] | None = None
                  ) -> dict:
    """Create train + multiple shifted test datasets.

    Each class is shifted in a different random direction, so increasing
    shift breaks the original decision boundaries and causes real
    misclassification and miscalibration.

    Args:
        seed: Random seed for full reproducibility.
        shift_magnitudes: List of shift amounts. Default: [0, 0.5, 1.0, 2.0, 3.0].

    Returns:
        Dict with keys:
            'train': (X_train, y_train)
            'test_shift_{mag}': (X_test, y_test) for each shift magnitude
            'centers': the original cluster centers
            'shift_magnitudes': list of shift values used
    """
    if shift_magnitudes is None:
        shift_magnitudes = [0.0, 0.5, 1.0, 2.0, 4.0]

    rng = np.random.default_rng(seed)
    centers = generate_cluster_centers(N_CLASSES, N_FEATURES, rng)

    # Per-class shift directions (fixed for all shift magnitudes within a seed)
    shift_dirs = _get_shift_directions(N_CLASSES, N_FEATURES, seed)

    # Training data: no shift
    X_train, y_train = generate_data(centers, N_SAMPLES_TRAIN, rng,
                                     shift_magnitude=0.0)

    result = {
        'train': (X_train, y_train),
        'centers': centers,
        'shift_magnitudes': shift_magnitudes,
    }

    for mag in shift_magnitudes:
        # Use a fresh but deterministic rng per shift to avoid correlation
        test_rng = np.random.default_rng(seed + int(mag * 1000) + 1)
        X_test, y_test = generate_data(centers, N_SAMPLES_TEST, test_rng,
                                       shift_magnitude=mag,
                                       shift_directions=shift_dirs)
        result[f'test_shift_{mag}'] = (X_test, y_test)

    return result
