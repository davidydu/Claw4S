"""Tests for synthetic data generation."""

import numpy as np
from src.data import (
    generate_gaussian_clusters,
    split_data,
    generate_shadow_data,
    N_SAMPLES,
    N_FEATURES,
    N_CLASSES,
    SEED,
)


def test_generate_gaussian_clusters_shape():
    """Generated data has expected shape."""
    X, y = generate_gaussian_clusters()
    assert X.shape == (N_SAMPLES, N_FEATURES)
    assert y.shape == (N_SAMPLES,)


def test_generate_gaussian_clusters_classes():
    """All expected classes are present."""
    X, y = generate_gaussian_clusters()
    unique_classes = set(y.tolist())
    assert unique_classes == set(range(N_CLASSES))


def test_generate_gaussian_clusters_reproducibility():
    """Same seed produces identical data."""
    X1, y1 = generate_gaussian_clusters(seed=SEED)
    X2, y2 = generate_gaussian_clusters(seed=SEED)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)


def test_generate_gaussian_clusters_different_seeds():
    """Different seeds produce different data."""
    X1, y1 = generate_gaussian_clusters(seed=42)
    X2, y2 = generate_gaussian_clusters(seed=99)
    assert not np.array_equal(X1, X2)


def test_split_data_sizes():
    """Split produces expected partition sizes."""
    X, y = generate_gaussian_clusters()
    X_tr, y_tr, X_te, y_te = split_data(X, y, train_fraction=0.5)
    assert len(X_tr) == N_SAMPLES // 2
    assert len(X_te) == N_SAMPLES - N_SAMPLES // 2


def test_split_data_no_overlap():
    """Train and test sets do not share samples (by content check)."""
    X, y = generate_gaussian_clusters()
    X_tr, y_tr, X_te, y_te = split_data(X, y, train_fraction=0.5)
    # Check that no row in train appears in test
    for i in range(len(X_tr)):
        diffs = np.abs(X_te - X_tr[i]).sum(axis=1)
        assert diffs.min() > 1e-10, f"Row {i} appears in both train and test"


def test_shadow_data_independence():
    """Shadow data is independent from target data."""
    X_target, y_target = generate_gaussian_clusters(seed=SEED)
    X_shadow, y_shadow = generate_shadow_data(shadow_idx=0, seed=SEED)
    assert not np.array_equal(X_target, X_shadow)


def test_shadow_data_different_indices():
    """Different shadow indices produce different data."""
    X0, y0 = generate_shadow_data(shadow_idx=0)
    X1, y1 = generate_shadow_data(shadow_idx=1)
    assert not np.array_equal(X0, X1)
