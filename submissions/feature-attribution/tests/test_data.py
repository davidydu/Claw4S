"""Tests for synthetic data generation."""

import numpy as np
from src.data import make_gaussian_clusters


def test_output_shapes():
    """Check that output shapes match specification."""
    X, y = make_gaussian_clusters(n_samples=100, n_features=8, n_classes=4, seed=42)
    assert X.shape == (100, 8), f"X shape {X.shape} != (100, 8)"
    assert y.shape == (100,), f"y shape {y.shape} != (100,)"


def test_dtypes():
    """Check output data types."""
    X, y = make_gaussian_clusters(n_samples=50, n_features=5, n_classes=3, seed=42)
    assert X.dtype == np.float32, f"X dtype {X.dtype} != float32"
    assert y.dtype == np.int64, f"y dtype {y.dtype} != int64"


def test_class_labels():
    """Check that all class labels are present."""
    n_classes = 5
    X, y = make_gaussian_clusters(n_samples=200, n_features=10, n_classes=n_classes, seed=42)
    unique = set(np.unique(y))
    expected = set(range(n_classes))
    assert unique == expected, f"Classes {unique} != {expected}"


def test_reproducibility():
    """Check that same seed produces identical data."""
    X1, y1 = make_gaussian_clusters(seed=42)
    X2, y2 = make_gaussian_clusters(seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_different_seeds_differ():
    """Check that different seeds produce different data."""
    X1, _ = make_gaussian_clusters(seed=42)
    X2, _ = make_gaussian_clusters(seed=99)
    assert not np.allclose(X1, X2), "Different seeds should produce different data"
