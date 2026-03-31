"""Synthetic data generation with spurious correlations.

Generates classification data with 10 genuine features drawn from class-conditional
Gaussians plus 1 shortcut feature that perfectly predicts labels in training but
is random at test time.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset


def generate_dataset(
    n_train: int = 2000,
    n_test: int = 1000,
    n_genuine: int = 10,
    seed: int = 42,
) -> dict:
    """Generate train/test datasets with a spurious shortcut feature.

    Training set: shortcut feature == label (perfect correlation).
    Test set: shortcut feature is random (no correlation).

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        n_genuine: Number of genuine (informative) features.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
        - train_dataset: TensorDataset(X, y)
        - test_with_shortcut: TensorDataset(X, y) — shortcut still correlated
        - test_without_shortcut: TensorDataset(X, y) — shortcut randomized
        - metadata: dict with generation parameters
    """
    rng = np.random.RandomState(seed)

    # --- Genuine features: class-conditional Gaussians ---
    # Class 0: features drawn from N(mu_0, I), Class 1: N(mu_1, I)
    # Means are slightly separated so the task is learnable but not trivial
    mu_0 = rng.randn(n_genuine) * 0.3
    mu_1 = mu_0 + rng.randn(n_genuine) * 0.5 + 0.5  # offset from class 0

    def _make_genuine(n: int, labels: np.ndarray) -> np.ndarray:
        X = np.zeros((n, n_genuine), dtype=np.float32)
        mask_0 = labels == 0
        mask_1 = labels == 1
        X[mask_0] = rng.randn(mask_0.sum(), n_genuine).astype(np.float32) + mu_0
        X[mask_1] = rng.randn(mask_1.sum(), n_genuine).astype(np.float32) + mu_1
        return X

    # --- Training set ---
    y_train = rng.randint(0, 2, size=n_train).astype(np.int64)
    X_genuine_train = _make_genuine(n_train, y_train)
    # Shortcut = label (perfect correlation in training)
    shortcut_train = y_train.astype(np.float32).reshape(-1, 1)
    X_train = np.concatenate([X_genuine_train, shortcut_train], axis=1)

    # --- Test set (with shortcut correlated) ---
    y_test = rng.randint(0, 2, size=n_test).astype(np.int64)
    X_genuine_test = _make_genuine(n_test, y_test)
    shortcut_test_corr = y_test.astype(np.float32).reshape(-1, 1)
    X_test_with = np.concatenate([X_genuine_test, shortcut_test_corr], axis=1)

    # --- Test set (shortcut randomized — no correlation) ---
    shortcut_test_rand = rng.randint(0, 2, size=n_test).astype(np.float32).reshape(-1, 1)
    X_test_without = np.concatenate([X_genuine_test, shortcut_test_rand], axis=1)

    def _to_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    return {
        "train_dataset": _to_dataset(X_train, y_train),
        "test_with_shortcut": _to_dataset(X_test_with, y_test),
        "test_without_shortcut": _to_dataset(X_test_without, y_test),
        "metadata": {
            "n_train": n_train,
            "n_test": n_test,
            "n_genuine": n_genuine,
            "n_total_features": n_genuine + 1,
            "shortcut_index": n_genuine,  # last column
            "seed": seed,
            "mu_0": mu_0.tolist(),
            "mu_1": mu_1.tolist(),
        },
    }
