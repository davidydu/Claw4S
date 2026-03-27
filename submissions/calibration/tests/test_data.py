"""Tests for synthetic data generation."""

import numpy as np
from src.data import (generate_cluster_centers, generate_data,
                      make_datasets, N_FEATURES, N_CLASSES,
                      N_SAMPLES_TRAIN, N_SAMPLES_TEST)


def test_cluster_centers_shape():
    """Cluster centers have correct shape."""
    rng = np.random.default_rng(42)
    centers = generate_cluster_centers(N_CLASSES, N_FEATURES, rng)
    assert centers.shape == (N_CLASSES, N_FEATURES)


def test_generate_data_shape():
    """Generated data has correct shape and label range."""
    rng = np.random.default_rng(42)
    centers = generate_cluster_centers(N_CLASSES, N_FEATURES, rng)
    X, y = generate_data(centers, 100, rng)
    assert X.shape == (100, N_FEATURES)
    assert y.shape == (100,)
    assert set(y.tolist()).issubset(set(range(N_CLASSES)))


def test_generate_data_with_shift():
    """Shifted data has different mean than unshifted."""
    rng = np.random.default_rng(42)
    centers = generate_cluster_centers(N_CLASSES, N_FEATURES, rng)

    rng1 = np.random.default_rng(99)
    X_no_shift, _ = generate_data(centers, 500, rng1, shift_magnitude=0.0)

    rng2 = np.random.default_rng(99)
    X_shifted, _ = generate_data(centers, 500, rng2, shift_magnitude=3.0)

    # The mean along axis 0 (first feature) should differ by ~3.0
    mean_diff = abs(X_shifted[:, 0].mean() - X_no_shift[:, 0].mean())
    assert mean_diff > 2.0, f"Expected shift > 2.0, got {mean_diff}"


def test_make_datasets_keys():
    """make_datasets returns all expected keys."""
    ds = make_datasets(seed=42)
    assert 'train' in ds
    assert 'centers' in ds
    assert 'shift_magnitudes' in ds
    for mag in ds['shift_magnitudes']:
        assert f'test_shift_{mag}' in ds


def test_make_datasets_shapes():
    """Training and test sets have correct shapes."""
    ds = make_datasets(seed=42)
    X_train, y_train = ds['train']
    assert X_train.shape == (N_SAMPLES_TRAIN, N_FEATURES)
    assert y_train.shape == (N_SAMPLES_TRAIN,)

    X_test, y_test = ds['test_shift_0.0']
    assert X_test.shape == (N_SAMPLES_TEST, N_FEATURES)
    assert y_test.shape == (N_SAMPLES_TEST,)


def test_reproducibility():
    """Same seed produces identical data."""
    ds1 = make_datasets(seed=42)
    ds2 = make_datasets(seed=42)
    np.testing.assert_array_equal(ds1['train'][0], ds2['train'][0])
    np.testing.assert_array_equal(ds1['train'][1], ds2['train'][1])


def test_different_seeds_differ():
    """Different seeds produce different data."""
    ds1 = make_datasets(seed=42)
    ds2 = make_datasets(seed=99)
    assert not np.array_equal(ds1['train'][0], ds2['train'][0])
