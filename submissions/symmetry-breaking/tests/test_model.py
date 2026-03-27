"""Tests for the SymmetricMLP model."""

import torch
from src.model import SymmetricMLP


def test_symmetric_init_identical_rows():
    """With epsilon=0, all fc1 weight rows should be identical."""
    model = SymmetricMLP(input_dim=10, hidden_dim=8, output_dim=5, epsilon=0.0)
    weights = model.fc1.weight.data
    # All rows should equal the first row
    for i in range(1, weights.size(0)):
        assert torch.allclose(weights[0], weights[i], atol=1e-7), (
            f"Row 0 and row {i} differ with epsilon=0"
        )


def test_symmetric_init_bias_zero():
    """fc1 bias should be initialized to zero."""
    model = SymmetricMLP(input_dim=10, hidden_dim=8, output_dim=5, epsilon=0.0)
    assert torch.allclose(model.fc1.bias.data, torch.zeros(8), atol=1e-7)


def test_perturbed_init_rows_differ():
    """With epsilon>0, fc1 weight rows should differ."""
    model = SymmetricMLP(input_dim=10, hidden_dim=8, output_dim=5, epsilon=0.1)
    weights = model.fc1.weight.data
    # At least some rows should differ
    diffs = 0
    for i in range(1, weights.size(0)):
        if not torch.allclose(weights[0], weights[i], atol=1e-7):
            diffs += 1
    assert diffs > 0, "No rows differ despite epsilon=0.1"


def test_forward_shape():
    """Forward pass should produce correct output shape."""
    model = SymmetricMLP(input_dim=20, hidden_dim=16, output_dim=10, epsilon=0.01)
    x = torch.randn(32, 20)
    out = model(x)
    assert out.shape == (32, 10), f"Expected (32, 10), got {out.shape}"


def test_reproducibility():
    """Same seed should produce identical weights."""
    m1 = SymmetricMLP(input_dim=10, hidden_dim=8, output_dim=5, epsilon=0.01, seed=42)
    m2 = SymmetricMLP(input_dim=10, hidden_dim=8, output_dim=5, epsilon=0.01, seed=42)
    assert torch.allclose(m1.fc1.weight.data, m2.fc1.weight.data, atol=1e-7)
    assert torch.allclose(m1.fc2.weight.data, m2.fc2.weight.data, atol=1e-7)


def test_different_seeds_differ():
    """Different seeds should produce different perturbations."""
    m1 = SymmetricMLP(input_dim=10, hidden_dim=8, output_dim=5, epsilon=0.01, seed=42)
    m2 = SymmetricMLP(input_dim=10, hidden_dim=8, output_dim=5, epsilon=0.01, seed=99)
    assert not torch.allclose(m1.fc1.weight.data, m2.fc1.weight.data, atol=1e-7)
