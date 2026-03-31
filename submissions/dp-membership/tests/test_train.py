"""Tests for model training."""

import torch
from torch.utils.data import TensorDataset

from src.dp_sgd import DPConfig
from src.train import train_model, evaluate_model


def _make_toy_dataset(n=50, d=5, c=3, seed=42):
    """Create a small toy dataset for testing."""
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    return TensorDataset(X, y)


def test_train_model_non_private():
    """Non-private training produces a model with decreasing loss."""
    ds = _make_toy_dataset()
    model, losses = train_model(
        train_dataset=ds, input_dim=5, num_classes=3,
        epochs=10, batch_size=16, lr=0.1, seed=42,
    )
    assert len(losses) == 10
    # Loss should generally decrease (first > last)
    assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.3f} -> {losses[-1]:.3f}"


def test_train_model_with_dp():
    """DP training completes and returns losses."""
    ds = _make_toy_dataset()
    dp_config = DPConfig(noise_multiplier=1.0, max_grad_norm=1.0)
    model, losses = train_model(
        train_dataset=ds, input_dim=5, num_classes=3,
        dp_config=dp_config, epochs=5, batch_size=16, lr=0.05, seed=42,
    )
    assert len(losses) == 5
    for loss in losses:
        assert loss >= 0


def test_evaluate_model():
    """Evaluation returns accuracy in [0,1] and non-negative loss."""
    ds = _make_toy_dataset()
    model, _ = train_model(
        train_dataset=ds, input_dim=5, num_classes=3,
        epochs=10, batch_size=16, seed=42,
    )
    acc, loss = evaluate_model(model, ds)
    assert 0.0 <= acc <= 1.0
    assert loss >= 0
