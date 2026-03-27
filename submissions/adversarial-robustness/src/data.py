"""
Synthetic dataset generators for adversarial robustness experiments.

Implements concentric circles and two-moons datasets using only numpy,
matching sklearn-style interfaces without the sklearn dependency.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def make_circles(n_samples: int = 1000, noise: float = 0.05,
                 factor: float = 0.5, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D concentric circles dataset.

    Args:
        n_samples: Total number of samples (split evenly between classes).
        noise: Standard deviation of Gaussian noise added to coordinates.
        factor: Scale factor between inner and outer circle (0 < factor < 1).
        seed: Random seed for reproducibility.

    Returns:
        X: array of shape (n_samples, 2) with 2D coordinates.
        y: array of shape (n_samples,) with binary labels {0, 1}.
    """
    rng = np.random.default_rng(seed)
    n_outer = n_samples // 2
    n_inner = n_samples - n_outer

    # Outer circle (label 0)
    theta_outer = rng.uniform(0, 2 * np.pi, n_outer)
    x_outer = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])

    # Inner circle (label 1)
    theta_inner = rng.uniform(0, 2 * np.pi, n_inner)
    x_inner = factor * np.column_stack([np.cos(theta_inner), np.sin(theta_inner)])

    X = np.vstack([x_outer, x_inner])
    y = np.concatenate([np.zeros(n_outer), np.ones(n_inner)])

    # Add noise
    X += rng.normal(0, noise, X.shape)

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm].astype(np.float32), y[perm].astype(np.int64)


def make_moons(n_samples: int = 1000, noise: float = 0.05,
               seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate 2D two-moons dataset.

    Args:
        n_samples: Total number of samples (split evenly between classes).
        noise: Standard deviation of Gaussian noise added to coordinates.
        seed: Random seed for reproducibility.

    Returns:
        X: array of shape (n_samples, 2) with 2D coordinates.
        y: array of shape (n_samples,) with binary labels {0, 1}.
    """
    rng = np.random.default_rng(seed)
    n_upper = n_samples // 2
    n_lower = n_samples - n_upper

    # Upper moon (label 0)
    theta_upper = np.linspace(0, np.pi, n_upper)
    x_upper = np.column_stack([np.cos(theta_upper), np.sin(theta_upper)])

    # Lower moon (label 1), shifted right and down
    theta_lower = np.linspace(0, np.pi, n_lower)
    x_lower = np.column_stack([1.0 - np.cos(theta_lower),
                                1.0 - np.sin(theta_lower) - 0.5])

    X = np.vstack([x_upper, x_lower])
    y = np.concatenate([np.zeros(n_upper), np.ones(n_lower)])

    # Add noise
    X += rng.normal(0, noise, X.shape)

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm].astype(np.float32), y[perm].astype(np.int64)


def make_dataloaders(dataset: str = "circles", n_samples: int = 2000,
                     noise: float = 0.05, test_fraction: float = 0.2,
                     batch_size: int = 256, seed: int = 42
                     ) -> tuple[DataLoader, DataLoader, torch.Tensor, torch.Tensor]:
    """Create train/test DataLoaders from a synthetic dataset.

    Args:
        dataset: One of "circles" or "moons".
        n_samples: Total number of samples.
        noise: Noise level for data generation.
        test_fraction: Fraction of data reserved for testing.
        batch_size: Batch size for DataLoaders.
        seed: Random seed.

    Returns:
        train_loader, test_loader, X_test_tensor, y_test_tensor
    """
    generators = {"circles": make_circles, "moons": make_moons}
    if dataset not in generators:
        raise ValueError(f"Unknown dataset: {dataset!r}. Choose from {list(generators.keys())}")

    X, y = generators[dataset](n_samples=n_samples, noise=noise, seed=seed)

    # Train/test split
    n_test = int(n_samples * test_fraction)
    n_train = n_samples - n_test

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, X_test_t, y_test_t
