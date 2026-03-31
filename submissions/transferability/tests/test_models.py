"""Tests for MLP model definitions."""

import torch
from src.models import MLP


def test_mlp_forward_shape():
    """MLP produces correct output shape."""
    model = MLP(input_dim=10, n_classes=5, hidden_width=32, n_hidden_layers=2)
    x = torch.randn(16, 10)
    out = model(x)
    assert out.shape == (16, 5), f"Expected (16, 5), got {out.shape}"


def test_mlp_4layer_forward():
    """4-layer MLP produces correct output shape."""
    model = MLP(input_dim=10, n_classes=5, hidden_width=64, n_hidden_layers=4)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 5), f"Expected (8, 5), got {out.shape}"


def test_mlp_param_count():
    """Param count is correct for a 2-layer MLP."""
    model = MLP(input_dim=10, n_classes=5, hidden_width=32, n_hidden_layers=2)
    # Layer 1: 10*32 + 32 = 352
    # Layer 2: 32*32 + 32 = 1056
    # Output:  32*5 + 5 = 165
    # Total: 352 + 1056 + 165 = 1573
    assert model.param_count() == 1573, f"Expected 1573, got {model.param_count()}"


def test_mlp_width_affects_params():
    """Wider MLPs have more parameters."""
    small = MLP(input_dim=10, n_classes=5, hidden_width=32, n_hidden_layers=2)
    large = MLP(input_dim=10, n_classes=5, hidden_width=256, n_hidden_layers=2)
    assert large.param_count() > small.param_count()


def test_mlp_depth_affects_params():
    """Deeper MLPs have more parameters."""
    shallow = MLP(input_dim=10, n_classes=5, hidden_width=64, n_hidden_layers=2)
    deep = MLP(input_dim=10, n_classes=5, hidden_width=64, n_hidden_layers=4)
    assert deep.param_count() > shallow.param_count()
