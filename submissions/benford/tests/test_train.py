"""Tests for training module."""

import torch

from src.data import generate_sine_data
from src.model import TinyMLP
from src.train import set_seed, train_model


def test_set_seed_reproducible():
    """Setting same seed produces same random values."""
    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_train_regression_loss_decreases():
    """Training on sine regression reduces loss."""
    set_seed(42)
    X_train, y_train, _, _ = generate_sine_data(n=200, seed=42)
    model = TinyMLP(d_in=1, d_hidden=32, d_out=1, n_hidden=2)

    snapshots, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        task_type="regression",
        epochs=200,
        lr=1e-3,
        snapshot_epochs=[0, 100, 200],
        seed=42,
    )

    # Loss should decrease
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_train_snapshots_saved():
    """Training saves snapshots at requested epochs."""
    set_seed(42)
    X_train, y_train, _, _ = generate_sine_data(n=100, seed=42)
    model = TinyMLP(d_in=1, d_hidden=16, d_out=1, n_hidden=2)

    snapshot_epochs = [0, 50, 100]
    snapshots, history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        task_type="regression",
        epochs=100,
        lr=1e-3,
        snapshot_epochs=snapshot_epochs,
        seed=42,
    )

    for epoch in snapshot_epochs:
        assert epoch in snapshots, f"Missing snapshot for epoch {epoch}"
        assert isinstance(snapshots[epoch], dict)


def test_train_snapshots_differ():
    """Snapshots at different epochs should have different weights."""
    set_seed(42)
    X_train, y_train, _, _ = generate_sine_data(n=100, seed=42)
    model = TinyMLP(d_in=1, d_hidden=16, d_out=1, n_hidden=2)

    snapshots, _ = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        task_type="regression",
        epochs=100,
        lr=1e-3,
        snapshot_epochs=[0, 100],
        seed=42,
    )

    # Weights at epoch 0 and 100 should differ
    for key in snapshots[0]:
        if not torch.equal(snapshots[0][key], snapshots[100][key]):
            return  # Found at least one different tensor
    assert False, "All weights identical between epoch 0 and 100"
