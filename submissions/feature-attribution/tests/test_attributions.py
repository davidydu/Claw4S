"""Tests for attribution methods."""

import numpy as np
import torch
from src.attributions import (
    vanilla_gradient,
    gradient_times_input,
    integrated_gradients,
    compute_all_attributions,
)
from src.models import MLP, train_model
from src.data import make_gaussian_clusters


def _trained_model():
    """Helper: return a small trained model and test input."""
    X, y = make_gaussian_clusters(n_samples=200, n_features=10, n_classes=5, seed=42)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)
    torch.manual_seed(42)
    model = MLP(n_features=10, n_classes=5, n_hidden=2, width=32)
    train_model(model, X_t, y_t, epochs=100, seed=42)
    model.eval()
    x_test = X_t[0:1]
    target = y_t[0].item()
    return model, x_test, target


def test_vanilla_gradient_shape():
    """Vanilla gradient returns correct shape."""
    model, x, target = _trained_model()
    attr = vanilla_gradient(model, x, target)
    assert attr.shape == (10,), f"Shape {attr.shape} != (10,)"
    assert np.all(attr >= 0), "Attributions should be non-negative (absolute values)"


def test_gradient_times_input_shape():
    """Gradient x Input returns correct shape."""
    model, x, target = _trained_model()
    attr = gradient_times_input(model, x, target)
    assert attr.shape == (10,), f"Shape {attr.shape} != (10,)"
    assert np.all(attr >= 0), "Attributions should be non-negative (absolute values)"


def test_integrated_gradients_shape():
    """Integrated gradients returns correct shape."""
    model, x, target = _trained_model()
    attr = integrated_gradients(model, x, target, n_steps=10)
    assert attr.shape == (10,), f"Shape {attr.shape} != (10,)"
    assert np.all(attr >= 0), "Attributions should be non-negative (absolute values)"


def test_compute_all_returns_three_methods():
    """compute_all_attributions returns all three methods."""
    model, x, target = _trained_model()
    attrs = compute_all_attributions(model, x, target, n_steps=10)
    expected_keys = {"vanilla_gradient", "gradient_x_input", "integrated_gradients"}
    assert set(attrs.keys()) == expected_keys, f"Keys: {attrs.keys()}"
    for key, val in attrs.items():
        assert val.shape == (10,), f"{key}: shape {val.shape} != (10,)"


def test_different_methods_give_different_results():
    """Attribution methods should generally give different rankings."""
    model, x, target = _trained_model()
    attrs = compute_all_attributions(model, x, target, n_steps=20)
    # At least one pair should differ (not be exactly identical)
    vg = attrs["vanilla_gradient"]
    gi = attrs["gradient_x_input"]
    ig = attrs["integrated_gradients"]
    any_differ = (
        not np.allclose(vg, gi) or
        not np.allclose(vg, ig) or
        not np.allclose(gi, ig)
    )
    assert any_differ, "All three methods should not be identical"
