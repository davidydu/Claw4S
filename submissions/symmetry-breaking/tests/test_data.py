"""Tests for the modular addition data generation."""

import torch
from src.data import generate_modular_addition_data, MODULUS


def test_data_shapes():
    """Generated data should have correct shapes."""
    x_train, y_train, x_test, y_test = generate_modular_addition_data()
    total = MODULUS * MODULUS
    assert x_train.size(0) + x_test.size(0) == total
    assert x_train.size(1) == 2 * MODULUS
    assert y_train.size(0) == x_train.size(0)
    assert y_test.size(0) == x_test.size(0)


def test_labels_in_range():
    """All labels should be in [0, MODULUS)."""
    _, y_train, _, y_test = generate_modular_addition_data()
    assert y_train.min() >= 0
    assert y_train.max() < MODULUS
    assert y_test.min() >= 0
    assert y_test.max() < MODULUS


def test_one_hot_encoding():
    """Each half of input should be a valid one-hot vector."""
    x_train, _, _, _ = generate_modular_addition_data()
    a_part = x_train[:, :MODULUS]
    b_part = x_train[:, MODULUS:]
    # Each row of a_part and b_part should sum to 1
    assert torch.allclose(a_part.sum(dim=1), torch.ones(x_train.size(0)))
    assert torch.allclose(b_part.sum(dim=1), torch.ones(x_train.size(0)))


def test_reproducibility():
    """Same seed should produce identical data."""
    x1, y1, _, _ = generate_modular_addition_data(seed=42)
    x2, y2, _, _ = generate_modular_addition_data(seed=42)
    assert torch.equal(x1, x2)
    assert torch.equal(y1, y2)


def test_train_test_split_ratio():
    """Train/test split should be approximately 80/20."""
    x_train, _, x_test, _ = generate_modular_addition_data()
    total = x_train.size(0) + x_test.size(0)
    train_ratio = x_train.size(0) / total
    assert 0.78 <= train_ratio <= 0.82, f"Train ratio {train_ratio:.2f} not near 0.8"
