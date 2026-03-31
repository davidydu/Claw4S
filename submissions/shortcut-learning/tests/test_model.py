"""Tests for the MLP model."""

import torch
from src.model import create_model, ShortcutMLP


def test_model_output_shape():
    """Model output should have shape (batch, 2) for binary classification."""
    model = create_model(input_dim=11, hidden_dim=64)
    X = torch.randn(16, 11)
    out = model(X)
    assert out.shape == (16, 2), f"Expected (16, 2), got {out.shape}"


def test_model_different_widths():
    """Model should work with different hidden dimensions."""
    for hd in [32, 64, 128]:
        model = create_model(input_dim=11, hidden_dim=hd)
        X = torch.randn(8, 11)
        out = model(X)
        assert out.shape == (8, 2), f"Failed for hidden_dim={hd}"


def test_model_is_mlp():
    """Model should be an instance of ShortcutMLP."""
    model = create_model()
    assert isinstance(model, ShortcutMLP)


def test_model_parameter_count_increases_with_width():
    """Wider models should have more parameters."""
    counts = {}
    for hd in [32, 64, 128]:
        model = create_model(input_dim=11, hidden_dim=hd)
        counts[hd] = sum(p.numel() for p in model.parameters())
    assert counts[32] < counts[64] < counts[128], \
        f"Parameter counts should increase with width: {counts}"
