"""Tests for src/data.py."""

import torch
from src.data import make_modular_addition_data, make_regression_data, MODULUS


def test_modular_data_shapes():
    """Modular addition data has correct shapes and dimensions."""
    data = make_modular_addition_data(modulus=MODULUS, seed=42)
    assert data["input_dim"] == 2 * MODULUS
    assert data["output_dim"] == MODULUS
    assert data["x_train"].shape[1] == 2 * MODULUS
    assert data["x_test"].shape[1] == 2 * MODULUS
    total = data["x_train"].shape[0] + data["x_test"].shape[0]
    assert total == MODULUS * MODULUS


def test_modular_data_labels_valid():
    """All labels are in [0, modulus)."""
    data = make_modular_addition_data(modulus=MODULUS, seed=42)
    assert (data["y_train"] >= 0).all()
    assert (data["y_train"] < MODULUS).all()
    assert (data["y_test"] >= 0).all()
    assert (data["y_test"] < MODULUS).all()


def test_modular_data_onehot():
    """Each input vector has exactly two 1s (one-hot encoded pair)."""
    data = make_modular_addition_data(modulus=MODULUS, seed=42)
    sums = data["x_train"].sum(dim=1)
    assert torch.allclose(sums, torch.full_like(sums, 2.0))


def test_modular_data_deterministic():
    """Same seed produces identical data."""
    d1 = make_modular_addition_data(modulus=MODULUS, seed=42)
    d2 = make_modular_addition_data(modulus=MODULUS, seed=42)
    assert torch.equal(d1["x_train"], d2["x_train"])
    assert torch.equal(d1["y_train"], d2["y_train"])


def test_regression_data_shapes():
    """Regression data has correct shapes."""
    data = make_regression_data(n_train=100, n_test=50, input_dim=10, seed=42)
    assert data["x_train"].shape == (100, 10)
    assert data["y_train"].shape == (100, 1)
    assert data["x_test"].shape == (50, 10)
    assert data["y_test"].shape == (50, 1)
    assert data["input_dim"] == 10
    assert data["output_dim"] == 1


def test_regression_data_deterministic():
    """Same seed produces identical regression data."""
    d1 = make_regression_data(seed=42)
    d2 = make_regression_data(seed=42)
    assert torch.equal(d1["x_train"], d2["x_train"])
    assert torch.equal(d1["y_train"], d2["y_train"])
