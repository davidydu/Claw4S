"""Tests for synthetic data generation."""

import numpy as np
import torch

from src.data import generate_gaussian_clusters, make_member_nonmember_split


def test_generate_gaussian_clusters_shapes():
    """Generated data has correct shapes."""
    X, y = generate_gaussian_clusters(n_samples=100, n_features=10, n_classes=5)
    assert X.shape == (100, 10)
    assert y.shape == (100,)


def test_generate_gaussian_clusters_classes():
    """All expected classes are present."""
    X, y = generate_gaussian_clusters(n_samples=500, n_features=10, n_classes=5)
    assert set(y) == {0, 1, 2, 3, 4}


def test_generate_gaussian_clusters_reproducible():
    """Same seed produces same data."""
    X1, y1 = generate_gaussian_clusters(seed=42)
    X2, y2 = generate_gaussian_clusters(seed=42)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_gaussian_clusters_different_seeds():
    """Different seeds produce different data."""
    X1, y1 = generate_gaussian_clusters(seed=42)
    X2, y2 = generate_gaussian_clusters(seed=99)
    assert not np.array_equal(X1, X2)


def test_member_nonmember_split():
    """Split produces correct sizes and types."""
    X, y = generate_gaussian_clusters(n_samples=100, n_features=10, n_classes=5)
    member_ds, nonmember_ds = make_member_nonmember_split(X, y, member_ratio=0.5)
    assert len(member_ds) == 50
    assert len(nonmember_ds) == 50
    assert isinstance(member_ds[0][0], torch.Tensor)


def test_member_nonmember_no_overlap():
    """Members and non-members do not share samples."""
    X, y = generate_gaussian_clusters(n_samples=100, n_features=5, n_classes=2)
    member_ds, nonmember_ds = make_member_nonmember_split(X, y, member_ratio=0.5)

    member_X = member_ds.tensors[0].numpy()
    nonmember_X = nonmember_ds.tensors[0].numpy()

    # Check no row in member_X appears in nonmember_X
    for row in member_X:
        distances = np.linalg.norm(nonmember_X - row, axis=1)
        assert distances.min() > 1e-6, "Found overlapping sample between member and non-member sets"
