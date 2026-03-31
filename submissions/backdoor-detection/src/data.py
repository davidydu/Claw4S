"""Synthetic data generation with backdoor injection."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate_clean_data(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian cluster classification data.

    Each class is a Gaussian cluster with a random mean vector and unit variance.
    Cluster centers are spaced to ensure reasonable separability.

    Args:
        n_samples: Total number of samples.
        n_features: Number of features per sample.
        n_classes: Number of target classes.
        seed: Random seed for reproducibility.

    Returns:
        X: Feature matrix of shape (n_samples, n_features).
        y: Label array of shape (n_samples,).
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes

    # Generate well-separated cluster centers
    centers = rng.randn(n_classes, n_features) * 3.0

    X_parts = []
    y_parts = []
    for cls_idx in range(n_classes):
        n = samples_per_class if cls_idx < n_classes - 1 else n_samples - samples_per_class * (n_classes - 1)
        X_cls = rng.randn(n, n_features) + centers[cls_idx]
        y_cls = np.full(n, cls_idx, dtype=np.int64)
        X_parts.append(X_cls)
        y_parts.append(y_cls)

    X = np.vstack(X_parts).astype(np.float32)
    y = np.concatenate(y_parts)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def inject_backdoor(
    X: np.ndarray,
    y: np.ndarray,
    poison_fraction: float = 0.1,
    target_class: int = 0,
    trigger_strength: float = 1.0,
    trigger_features: tuple[int, ...] = (0, 1, 2),
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Inject backdoor trigger into a subset of training data.

    Selects a random subset of non-target-class samples, applies the trigger
    pattern (setting specified features to fixed values scaled by trigger_strength),
    and relabels them to the target class.

    Args:
        X: Clean feature matrix.
        y: Clean labels.
        poison_fraction: Fraction of total samples to poison.
        target_class: Target class for backdoor relabeling.
        trigger_strength: Magnitude of trigger pattern values.
        trigger_features: Feature indices to modify as the trigger.
        seed: Random seed for reproducibility.

    Returns:
        X_poisoned: Feature matrix with backdoor injected.
        y_poisoned: Labels with poisoned samples relabeled.
        poison_mask: Boolean array indicating which samples are poisoned.
    """
    rng = np.random.RandomState(seed)
    n_poison = max(1, int(len(X) * poison_fraction))

    # Only poison samples not already in target class
    non_target_idx = np.where(y != target_class)[0]
    if n_poison > len(non_target_idx):
        n_poison = len(non_target_idx)

    chosen = rng.choice(non_target_idx, size=n_poison, replace=False)

    X_poisoned = X.copy()
    y_poisoned = y.copy()
    poison_mask = np.zeros(len(X), dtype=bool)

    # Apply trigger: set specified features to fixed values
    for feat_idx in trigger_features:
        X_poisoned[chosen, feat_idx] = trigger_strength

    # Relabel poisoned samples to target class
    y_poisoned[chosen] = target_class
    poison_mask[chosen] = True

    return X_poisoned, y_poisoned, poison_mask


def make_datasets(
    X: np.ndarray, y: np.ndarray
) -> TensorDataset:
    """Convert numpy arrays to a PyTorch TensorDataset.

    Args:
        X: Feature matrix.
        y: Labels.

    Returns:
        A TensorDataset containing feature and label tensors.
    """
    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).long()
    return TensorDataset(X_tensor, y_tensor)
