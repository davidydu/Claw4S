"""Tests for the TwoLayerMLP model."""

import torch
from src.model import TwoLayerMLP


def test_model_forward_shape():
    """Output shape matches (batch_size, output_dim)."""
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
    x = torch.randn(8, 10)
    out = model(x)
    assert out.shape == (8, 5)


def test_model_parameter_count():
    """Parameter count matches expected value for 2-layer MLP."""
    # fc1: 10*32 + 32 = 352, fc2: 32*5 + 5 = 165, total = 517
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
    assert model.count_parameters() == 517


def test_model_nonzero_params_equals_total_before_pruning():
    """Before pruning, all parameters are non-zero (with high probability)."""
    torch.manual_seed(42)
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
    total = model.count_parameters()
    nonzero = model.count_nonzero_parameters()
    # With random init, essentially all params should be nonzero
    assert nonzero >= total - 5  # allow a few accidental zeros


def test_model_deterministic_with_seed():
    """Same seed produces identical model weights."""
    torch.manual_seed(42)
    m1 = TwoLayerMLP(10, 32, 5)
    torch.manual_seed(42)
    m2 = TwoLayerMLP(10, 32, 5)
    for p1, p2 in zip(m1.parameters(), m2.parameters()):
        assert torch.equal(p1, p2)


def test_model_baseline_config():
    """The modular-arithmetic baseline config has the expected parameter count."""
    model = TwoLayerMLP(input_dim=194, hidden_dim=128, output_dim=97)
    assert model.count_parameters() == 37473
