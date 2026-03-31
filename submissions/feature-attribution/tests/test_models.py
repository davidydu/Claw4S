"""Tests for MLP model and training."""

import torch
import numpy as np
from src.models import MLP, train_model
from src.data import make_gaussian_clusters


def test_mlp_output_shape():
    """MLP forward pass produces correct output shape."""
    model = MLP(n_features=10, n_classes=5, n_hidden=2, width=32)
    x = torch.randn(16, 10)
    out = model(x)
    assert out.shape == (16, 5), f"Output shape {out.shape} != (16, 5)"


def test_mlp_depths():
    """MLPs with different depths have different parameter counts."""
    params = []
    for depth in [1, 2, 4]:
        model = MLP(n_features=10, n_classes=5, n_hidden=depth, width=32)
        n_params = sum(p.numel() for p in model.parameters())
        params.append(n_params)
    # Deeper models should have more parameters
    assert params[0] < params[1] < params[2], f"Params should increase: {params}"


def test_training_reduces_loss():
    """Training should reduce loss substantially."""
    X, y = make_gaussian_clusters(n_samples=200, n_features=10, n_classes=5, seed=42)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    model = MLP(n_features=10, n_classes=5, n_hidden=2, width=64)
    losses = train_model(model, X_t, y_t, epochs=50, seed=42)

    assert losses[-1] < losses[0], "Final loss should be lower than initial loss"
    assert losses[-1] < 1.0, f"Final loss {losses[-1]:.3f} should be < 1.0"


def test_training_reproducibility():
    """Same seed should produce same training trajectory."""
    X, y = make_gaussian_clusters(n_samples=100, n_features=10, n_classes=5, seed=42)
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y)

    torch.manual_seed(42)
    m1 = MLP(n_features=10, n_classes=5, n_hidden=1, width=32)
    l1 = train_model(m1, X_t, y_t, epochs=20, seed=42)

    torch.manual_seed(42)
    m2 = MLP(n_features=10, n_classes=5, n_hidden=1, width=32)
    l2 = train_model(m2, X_t, y_t, epochs=20, seed=42)

    np.testing.assert_allclose(l1, l2, rtol=1e-5)
