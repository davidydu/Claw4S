"""Tests for src/models.py."""

import torch
import pytest
from src.models import ReLUMLP, create_model


def test_relu_mlp_output_shape():
    """Model output shape matches expected dimensions."""
    model = ReLUMLP(input_dim=10, hidden_dim=32, output_dim=5)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 5)


def test_relu_mlp_hidden_shape():
    """Hidden activations have correct shape after forward pass."""
    model = ReLUMLP(input_dim=10, hidden_dim=32, output_dim=5)
    x = torch.randn(8, 10)
    _ = model(x)
    hidden = model.get_last_hidden()
    assert hidden.shape == (8, 32)


def test_relu_mlp_hidden_nonneg():
    """Hidden activations are non-negative (ReLU output)."""
    model = ReLUMLP(input_dim=10, hidden_dim=64, output_dim=5)
    x = torch.randn(16, 10)
    _ = model(x)
    hidden = model.get_last_hidden()
    assert (hidden >= 0).all()


def test_relu_mlp_no_forward_raises():
    """Getting hidden activations before forward pass raises RuntimeError."""
    model = ReLUMLP(input_dim=10, hidden_dim=32, output_dim=5)
    with pytest.raises(RuntimeError, match="No forward pass"):
        model.get_last_hidden()


def test_create_model_deterministic():
    """create_model produces identical weights with same seed."""
    m1 = create_model(10, 32, 5, seed=42)
    m2 = create_model(10, 32, 5, seed=42)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_create_model_different_seeds():
    """create_model produces different weights with different seeds."""
    m1 = create_model(10, 32, 5, seed=42)
    m2 = create_model(10, 32, 5, seed=99)
    params_equal = all(
        torch.equal(p1, p2) for p1, p2 in zip(m1.parameters(), m2.parameters())
    )
    assert not params_equal
