"""Tests for data generation."""

import torch
from src.data import generate_modular_data, generate_regression_data


def test_modular_data_shapes():
    """Modular data has correct shapes for mod=97."""
    X_train, y_train, X_test, y_test = generate_modular_data(mod=97, seed=42)
    total = 97 * 97
    train_n = int(0.8 * total)
    test_n = total - train_n
    assert X_train.shape == (train_n, 194)
    assert y_train.shape == (train_n,)
    assert X_test.shape == (test_n, 194)
    assert y_test.shape == (test_n,)


def test_modular_data_labels_in_range():
    """All labels are in [0, mod)."""
    X_train, y_train, X_test, y_test = generate_modular_data(mod=97, seed=42)
    assert y_train.min() >= 0
    assert y_train.max() < 97
    assert y_test.min() >= 0
    assert y_test.max() < 97


def test_modular_data_one_hot():
    """Each input row has exactly 2 nonzero entries (one-hot encoding)."""
    X_train, _, _, _ = generate_modular_data(mod=97, seed=42)
    nonzero_counts = (X_train != 0).sum(dim=1)
    assert (nonzero_counts == 2).all()


def test_modular_data_reproducible():
    """Same seed produces identical data."""
    d1 = generate_modular_data(mod=97, seed=42)
    d2 = generate_modular_data(mod=97, seed=42)
    for t1, t2 in zip(d1, d2):
        assert torch.equal(t1, t2)


def test_regression_data_shapes():
    """Regression data has correct shapes."""
    X_train, y_train, X_test, y_test = generate_regression_data(
        n_samples=200, n_features=20, seed=42
    )
    assert X_train.shape == (160, 20)
    assert y_train.shape == (160,)
    assert X_test.shape == (40, 20)
    assert y_test.shape == (40,)


def test_regression_data_reproducible():
    """Same seed produces identical regression data."""
    d1 = generate_regression_data(seed=42)
    d2 = generate_regression_data(seed=42)
    for t1, t2 in zip(d1, d2):
        assert torch.equal(t1, t2)
