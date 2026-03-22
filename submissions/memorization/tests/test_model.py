# tests/test_model.py
"""Tests for the MLP model."""

import torch
from src.model import MLP, count_parameters


def test_mlp_output_shape():
    """MLP produces correct output shape."""
    model = MLP(input_dim=20, hidden_dim=64, num_classes=10)
    x = torch.randn(32, 20)
    out = model(x)
    assert out.shape == (32, 10)


def test_mlp_single_sample():
    """MLP works with a single sample."""
    model = MLP(input_dim=5, hidden_dim=10, num_classes=3)
    x = torch.randn(1, 5)
    out = model(x)
    assert out.shape == (1, 3)


def test_count_parameters():
    """Parameter count matches expected formula."""
    model = MLP(input_dim=20, hidden_dim=10, num_classes=10)
    n_params = count_parameters(model)
    # Layer 1: 20*10 + 10 = 210
    # Layer 2: 10*10 + 10 = 110
    # Total: 320
    expected = 20 * 10 + 10 + 10 * 10 + 10
    assert n_params == expected, f"Expected {expected}, got {n_params}"


def test_count_parameters_scales_with_width():
    """Wider models have more parameters."""
    small = MLP(hidden_dim=10)
    large = MLP(hidden_dim=100)
    assert count_parameters(large) > count_parameters(small)


def test_parameter_formula():
    """Verify parameter formula: h*(d + n_classes) + (h + n_classes)."""
    for h in [5, 10, 20, 40, 80]:
        d, c = 20, 10
        model = MLP(input_dim=d, hidden_dim=h, num_classes=c)
        expected = h * (d + c) + (h + c)
        actual = count_parameters(model)
        assert actual == expected, f"h={h}: expected {expected}, got {actual}"
