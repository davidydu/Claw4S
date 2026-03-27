"""Tests for the MLP model."""

import torch

from src.model import MLP


def test_mlp_forward_shape():
    """MLP produces correct output shape."""
    model = MLP(input_dim=10, hidden_dim=64, num_classes=5)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 5)


def test_mlp_single_sample():
    """MLP handles single sample."""
    model = MLP(input_dim=10, hidden_dim=32, num_classes=3)
    x = torch.randn(1, 10)
    out = model(x)
    assert out.shape == (1, 3)


def test_mlp_reproducible():
    """Same seed produces same initial weights."""
    torch.manual_seed(42)
    m1 = MLP(input_dim=10, hidden_dim=64, num_classes=5)
    w1 = list(m1.parameters())[0].data.clone()

    torch.manual_seed(42)
    m2 = MLP(input_dim=10, hidden_dim=64, num_classes=5)
    w2 = list(m2.parameters())[0].data.clone()

    assert torch.allclose(w1, w2)
