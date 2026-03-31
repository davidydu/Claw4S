"""Tests for MLP model training and inference."""

import numpy as np
import torch
from src.models import (
    TwoLayerMLP,
    train_model,
    get_predictions,
    compute_accuracy,
    create_and_train_model,
    set_seed,
    SEED,
)
from src.data import generate_gaussian_clusters, split_data


def test_mlp_forward_shape():
    """MLP forward pass produces correct output shape."""
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
    x = torch.randn(16, 10)
    out = model(x)
    assert out.shape == (16, 5)


def test_mlp_parameter_count():
    """MLP has expected number of parameters."""
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
    n_params = sum(p.numel() for p in model.parameters())
    # Layer 1: 10*32 + 32 = 352, Layer 2: 32*5 + 5 = 165, Total = 517
    assert n_params == 517


def test_train_model_loss_decreases():
    """Training loss decreases over epochs."""
    X, y = generate_gaussian_clusters(n_samples=100, seed=SEED)
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
    losses = train_model(model, X, y, epochs=50, seed=SEED)
    assert losses[-1] < losses[0], "Loss should decrease during training"


def test_get_predictions_shape():
    """Predictions have correct shape and are valid probabilities."""
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)
    X = np.random.randn(20, 10).astype(np.float32)
    probs = get_predictions(model, X)
    assert probs.shape == (20, 5)
    # Check valid probability distributions
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert (probs >= 0).all()


def test_compute_accuracy_range():
    """Accuracy is between 0 and 1."""
    X, y = generate_gaussian_clusters(n_samples=100, seed=SEED)
    X_tr, y_tr, X_te, y_te = split_data(X, y, seed=SEED)
    model, _ = create_and_train_model(
        hidden_width=32, X_train=X_tr, y_train=y_tr, seed=SEED
    )
    acc = compute_accuracy(model, X_te, y_te)
    assert 0.0 <= acc <= 1.0


def test_training_reproducibility():
    """Same seed produces identical training results."""
    X, y = generate_gaussian_clusters(n_samples=100, seed=SEED)

    model1, losses1 = create_and_train_model(
        hidden_width=32, X_train=X, y_train=y, seed=SEED
    )
    model2, losses2 = create_and_train_model(
        hidden_width=32, X_train=X, y_train=y, seed=SEED
    )

    probs1 = get_predictions(model1, X)
    probs2 = get_predictions(model2, X)
    np.testing.assert_allclose(probs1, probs2, atol=1e-5)


def test_trained_model_above_random():
    """A trained model should achieve above random accuracy (>20% for 5 classes)."""
    X, y = generate_gaussian_clusters(n_samples=200, seed=SEED)
    X_tr, y_tr, X_te, y_te = split_data(X, y, seed=SEED)
    model, _ = create_and_train_model(
        hidden_width=64, X_train=X_tr, y_train=y_tr, seed=SEED, epochs=100
    )
    acc = compute_accuracy(model, X_te, y_te)
    assert acc > 0.3, f"Expected accuracy > 0.3, got {acc}"
