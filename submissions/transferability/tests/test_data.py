"""Tests for synthetic data generation."""

import torch
from src.data import make_gaussian_clusters


def test_dataset_shape():
    """Dataset has correct number of samples and features."""
    ds = make_gaussian_clusters(n_samples=100, n_features=10, n_classes=5, seed=42)
    X, y = ds.tensors
    assert X.shape == (100, 10), f"Expected (100, 10), got {X.shape}"
    assert y.shape == (100,), f"Expected (100,), got {y.shape}"


def test_dataset_types():
    """Dataset tensors have correct dtypes."""
    ds = make_gaussian_clusters(n_samples=50, n_features=5, n_classes=3, seed=42)
    X, y = ds.tensors
    assert X.dtype == torch.float32, f"Expected float32, got {X.dtype}"
    assert y.dtype == torch.int64, f"Expected int64, got {y.dtype}"


def test_dataset_classes():
    """All expected classes are present."""
    ds = make_gaussian_clusters(n_samples=100, n_features=5, n_classes=5, seed=42)
    _, y = ds.tensors
    unique = torch.unique(y)
    assert len(unique) == 5, f"Expected 5 classes, got {len(unique)}"
    assert set(unique.tolist()) == {0, 1, 2, 3, 4}


def test_dataset_reproducibility():
    """Same seed produces identical datasets."""
    ds1 = make_gaussian_clusters(n_samples=50, n_features=5, n_classes=3, seed=99)
    ds2 = make_gaussian_clusters(n_samples=50, n_features=5, n_classes=3, seed=99)
    assert torch.equal(ds1.tensors[0], ds2.tensors[0])
    assert torch.equal(ds1.tensors[1], ds2.tensors[1])


def test_different_seeds_differ():
    """Different seeds produce different datasets."""
    ds1 = make_gaussian_clusters(n_samples=50, n_features=5, n_classes=3, seed=1)
    ds2 = make_gaussian_clusters(n_samples=50, n_features=5, n_classes=3, seed=2)
    assert not torch.equal(ds1.tensors[0], ds2.tensors[0])
