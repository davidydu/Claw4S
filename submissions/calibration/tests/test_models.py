"""Tests for MLP models."""

import torch
import numpy as np
from src.models import TwoLayerMLP, train_model, predict_proba
from src.data import N_FEATURES, N_CLASSES


def test_model_output_shape():
    """Model outputs correct shape."""
    model = TwoLayerMLP(N_FEATURES, 32, N_CLASSES)
    X = torch.randn(10, N_FEATURES)
    out = model(X)
    assert out.shape == (10, N_CLASSES)


def test_model_param_count():
    """Parameter count matches expected formula."""
    # 2-layer MLP: (input * hidden + hidden) + (hidden * output + output)
    hidden = 64
    expected = (N_FEATURES * hidden + hidden) + (hidden * N_CLASSES + N_CLASSES)
    model = TwoLayerMLP(N_FEATURES, hidden, N_CLASSES)
    actual = sum(p.numel() for p in model.parameters())
    assert actual == expected, f"Expected {expected}, got {actual}"


def test_training_reduces_loss():
    """Training should reduce loss."""
    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn(100, N_FEATURES)
    y = torch.randint(0, N_CLASSES, (100,))

    model = TwoLayerMLP(N_FEATURES, 32, N_CLASSES)
    losses = train_model(model, X, y, epochs=50)
    assert losses[-1] < losses[0], "Final loss should be less than initial loss"


def test_predict_proba_sums_to_one():
    """Predicted probabilities sum to 1."""
    torch.manual_seed(42)
    model = TwoLayerMLP(N_FEATURES, 32, N_CLASSES)
    X = torch.randn(20, N_FEATURES)
    probs, preds = predict_proba(model, X)

    # Check probabilities sum to 1
    sums = probs.sum(dim=1)
    assert torch.allclose(sums, torch.ones(20), atol=1e-5)

    # Check predictions are valid class indices
    assert (preds >= 0).all() and (preds < N_CLASSES).all()


def test_predict_proba_shapes():
    """predict_proba returns correct shapes."""
    torch.manual_seed(42)
    model = TwoLayerMLP(N_FEATURES, 64, N_CLASSES)
    X = torch.randn(15, N_FEATURES)
    probs, preds = predict_proba(model, X)
    assert probs.shape == (15, N_CLASSES)
    assert preds.shape == (15,)
