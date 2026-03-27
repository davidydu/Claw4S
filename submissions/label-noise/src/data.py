"""Synthetic classification data generation with controllable label noise."""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def make_gaussian_clusters(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    seed: int = 42,
    cluster_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic Gaussian cluster data.

    Each class is a Gaussian blob with its centroid placed on a vertex of a
    regular simplex in ``n_features``-dimensional space so that inter-class
    distances are equal.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) with integer labels in [0, n_classes)
    """
    rng = np.random.RandomState(seed)
    samples_per_class = n_samples // n_classes
    remainder = n_samples - samples_per_class * n_classes

    # Place centroids: use random orthogonal directions scaled so centroids
    # are well-separated relative to cluster_std.
    centroid_scale = 3.0 * cluster_std
    raw = rng.randn(n_classes, n_features)
    # QR to get orthogonal directions (works when n_classes <= n_features)
    if n_classes <= n_features:
        q, _ = np.linalg.qr(raw.T)
        centroids = q[:, :n_classes].T * centroid_scale
    else:
        centroids = raw / np.linalg.norm(raw, axis=1, keepdims=True) * centroid_scale

    X_parts, y_parts = [], []
    for c in range(n_classes):
        n_c = samples_per_class + (1 if c < remainder else 0)
        X_c = rng.randn(n_c, n_features) * cluster_std + centroids[c]
        y_c = np.full(n_c, c, dtype=np.int64)
        X_parts.append(X_c)
        y_parts.append(y_c)

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def inject_label_noise(
    y: np.ndarray,
    noise_frac: float,
    n_classes: int,
    seed: int = 42,
) -> np.ndarray:
    """Randomly flip a fraction of labels to a *different* class.

    Parameters
    ----------
    y : ndarray of integer labels
    noise_frac : float in [0, 1]
    n_classes : int
    seed : int

    Returns
    -------
    y_noisy : ndarray with ``noise_frac`` of labels changed
    """
    if noise_frac < 0.0 or noise_frac > 1.0:
        raise ValueError(f"noise_frac must be in [0, 1], got {noise_frac}")
    if noise_frac == 0.0:
        return y.copy()

    rng = np.random.RandomState(seed)
    y_noisy = y.copy()
    n = len(y)
    n_flip = int(round(n * noise_frac))
    flip_idx = rng.choice(n, size=n_flip, replace=False)

    for i in flip_idx:
        # Pick a different class uniformly
        candidates = [c for c in range(n_classes) if c != y[i]]
        y_noisy[i] = rng.choice(candidates)

    return y_noisy


def build_datasets(
    n_samples: int = 500,
    n_features: int = 10,
    n_classes: int = 5,
    noise_frac: float = 0.0,
    seed: int = 42,
    train_frac: float = 0.7,
) -> tuple[TensorDataset, TensorDataset, np.ndarray]:
    """Build train/test TensorDatasets with optional label noise on train set.

    Returns
    -------
    train_ds : TensorDataset (with noisy labels)
    test_ds  : TensorDataset (with clean labels)
    y_train_clean : ndarray of clean training labels (for diagnostics)
    """
    X, y = make_gaussian_clusters(
        n_samples=n_samples,
        n_features=n_features,
        n_classes=n_classes,
        seed=seed,
    )

    n_train = int(n_samples * train_frac)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train_clean, y_test = y[:n_train], y[n_train:]

    # Inject noise only into training labels; use a seed derived from the
    # base seed so that different noise fractions get different flips but
    # the *same* underlying data split.
    y_train_noisy = inject_label_noise(
        y_train_clean, noise_frac, n_classes, seed=seed + 1000
    )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train_noisy, dtype=torch.long),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long),
    )

    return train_ds, test_ds, y_train_clean
