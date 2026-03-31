# src/data.py
"""Synthetic data generation for memorization experiments."""

import numpy as np
import torch


def generate_dataset(
    n_train: int = 200,
    n_test: int = 50,
    d: int = 20,
    n_classes: int = 10,
    seed: int = 42,
    label_type: str = "random",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate synthetic classification dataset.

    Args:
        n_train: Number of training samples.
        n_test: Number of test samples.
        d: Feature dimensionality.
        n_classes: Number of classes.
        seed: Random seed for reproducibility.
        label_type: "random" for random labels, "structured" for cluster-based labels.

    Returns:
        (X_train, y_train, X_test, y_test) as torch tensors.

    Raises:
        ValueError: If label_type is not "random" or "structured".
    """
    if label_type not in ("random", "structured"):
        raise ValueError(f"label_type must be 'random' or 'structured', got '{label_type}'")

    rng = np.random.RandomState(seed)
    n_total = n_train + n_test

    # Generate features: X ~ N(0, 1)
    X = rng.randn(n_total, d).astype(np.float32)

    if label_type == "random":
        # Random labels: uniform over classes
        y = rng.randint(0, n_classes, size=n_total)
    else:
        # Structured labels: k-means cluster assignments
        # Use a simple deterministic clustering: assign to nearest centroid
        # Generate centroids from the data itself
        centroid_indices = rng.choice(n_total, size=n_classes, replace=False)
        centroids = X[centroid_indices]

        # Assign each point to nearest centroid
        # distances shape: (n_total, n_classes)
        diffs = X[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sum(diffs ** 2, axis=2)
        y = np.argmin(distances, axis=1)

    # Split into train/test
    X_train = torch.from_numpy(X[:n_train])
    y_train = torch.from_numpy(y[:n_train].astype(np.int64))
    X_test = torch.from_numpy(X[n_train:])
    y_test = torch.from_numpy(y[n_train:].astype(np.int64))

    return X_train, y_train, X_test, y_test
