"""Tests for the TinyMLP model."""

import torch
from src.model import TinyMLP


def test_forward_shape_classification():
    """Verify output shape for classification task."""
    model = TinyMLP(input_dim=10, hidden_dim=32, output_dim=5)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 5)


def test_forward_shape_regression():
    """Verify output shape for regression task."""
    model = TinyMLP(input_dim=3, hidden_dim=16, output_dim=1)
    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 1)


def test_get_weight_matrices():
    """Verify get_weight_matrices returns 3 layers with correct shapes."""
    model = TinyMLP(input_dim=10, hidden_dim=32, output_dim=5)
    weights = model.get_weight_matrices()

    assert len(weights) == 3
    names = [name for name, _ in weights]
    assert names == ["fc1", "fc2", "fc3"]

    # fc1: 32 x 10
    assert weights[0][1].shape == (32, 10)
    # fc2: 32 x 32
    assert weights[1][1].shape == (32, 32)
    # fc3: 5 x 32
    assert weights[2][1].shape == (5, 32)


def test_weight_matrices_detached():
    """Verify weight matrices are detached from computation graph."""
    model = TinyMLP(input_dim=10, hidden_dim=32, output_dim=5)
    weights = model.get_weight_matrices()
    for _, W in weights:
        assert not W.requires_grad
