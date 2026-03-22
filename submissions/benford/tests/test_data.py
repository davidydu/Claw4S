"""Tests for data generation module."""

import torch

from src.data import generate_modular_data, generate_sine_data


def test_modular_data_shapes():
    """Modular arithmetic data has correct shapes."""
    X_train, y_train, X_test, y_test = generate_modular_data(p=97, seed=42)
    total = X_train.shape[0] + X_test.shape[0]
    assert total == 97 * 97, f"Expected {97*97} total, got {total}"
    assert X_train.shape[1] == 2
    assert X_test.shape[1] == 2
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]


def test_modular_data_values():
    """Modular arithmetic inputs in [0,1] and labels in [0, p)."""
    X_train, y_train, X_test, y_test = generate_modular_data(p=97, seed=42)
    assert X_train.min() >= 0.0
    assert X_train.max() <= 1.0
    assert y_train.min() >= 0
    assert y_train.max() < 97


def test_modular_data_reproducible():
    """Same seed produces same data."""
    d1 = generate_modular_data(p=97, seed=42)
    d2 = generate_modular_data(p=97, seed=42)
    assert torch.equal(d1[0], d2[0])
    assert torch.equal(d1[1], d2[1])


def test_sine_data_shapes():
    """Sine regression data has correct shapes."""
    X_train, y_train, X_test, y_test = generate_sine_data(n=1000, seed=42)
    total = X_train.shape[0] + X_test.shape[0]
    assert total == 1000
    assert X_train.shape[1] == 1
    assert y_train.shape[1] == 1


def test_sine_data_values():
    """Sine regression outputs in [-1, 1]."""
    _, y_train, _, y_test = generate_sine_data(n=1000, seed=42)
    all_y = torch.cat([y_train, y_test])
    assert all_y.min() >= -1.0
    assert all_y.max() <= 1.0


def test_sine_data_reproducible():
    """Same seed produces same data."""
    d1 = generate_sine_data(n=100, seed=42)
    d2 = generate_sine_data(n=100, seed=42)
    assert torch.equal(d1[0], d2[0])
    assert torch.equal(d1[1], d2[1])
