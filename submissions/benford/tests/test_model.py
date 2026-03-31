"""Tests for MLP model module."""

import torch

from src.model import TinyMLP


def test_forward_shape_classification():
    """Forward pass produces correct output shape for classification."""
    model = TinyMLP(d_in=2, d_hidden=64, d_out=97, n_hidden=2)
    x = torch.randn(32, 2)
    out = model(x)
    assert out.shape == (32, 97)


def test_forward_shape_regression():
    """Forward pass produces correct output shape for regression."""
    model = TinyMLP(d_in=1, d_hidden=64, d_out=1, n_hidden=2)
    x = torch.randn(32, 1)
    out = model(x)
    assert out.shape == (32, 1)


def test_parameter_count():
    """Model has expected number of parameters."""
    model = TinyMLP(d_in=2, d_hidden=64, d_out=97, n_hidden=2)
    n_params = model.count_parameters()
    # Layer 0: 2*64 + 64 = 192
    # Layer 1: 64*64 + 64 = 4160
    # Layer 2: 64*97 + 97 = 6305
    expected = 192 + 4160 + 6305
    assert n_params == expected, f"Expected {expected}, got {n_params}"


def test_get_weight_layers():
    """get_weight_layers returns weight tensors (not biases)."""
    model = TinyMLP(d_in=2, d_hidden=64, d_out=10, n_hidden=2)
    weights = model.get_weight_layers()
    assert len(weights) == 3  # 2 hidden + 1 output
    for name, tensor in weights.items():
        assert "weight" in name
        assert "bias" not in name


def test_different_sizes():
    """Model scales correctly with hidden size."""
    small = TinyMLP(d_in=2, d_hidden=64, d_out=10, n_hidden=2)
    large = TinyMLP(d_in=2, d_hidden=128, d_out=10, n_hidden=2)
    assert large.count_parameters() > small.count_parameters()
