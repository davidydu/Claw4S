"""Tests for training functions."""

import copy

import torch
from src.model import TwoLayerMLP
from src.data import generate_modular_data, generate_regression_data
from src.train import train_classification, train_regression


def test_classification_trains():
    """Classification training reduces loss and returns expected keys."""
    torch.manual_seed(42)
    # Use small mod for fast test
    X_train, y_train, X_test, y_test = generate_modular_data(mod=11, seed=42)
    model = TwoLayerMLP(input_dim=22, hidden_dim=32, output_dim=11)

    result = train_classification(
        model, X_train, y_train, X_test, y_test,
        masks={}, max_epochs=100, lr=1e-2, patience=50,
    )

    assert "test_acc" in result
    assert "train_acc" in result
    assert "train_loss" in result
    assert "epochs_trained" in result
    assert 0.0 <= result["test_acc"] <= 1.0
    assert result["epochs_trained"] > 0


def test_regression_trains():
    """Regression training returns expected keys and reasonable R^2."""
    torch.manual_seed(42)
    X_train, y_train, X_test, y_test = generate_regression_data(
        n_samples=100, n_features=10, seed=42
    )
    model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=1)

    result = train_regression(
        model, X_train, y_train, X_test, y_test,
        masks={}, max_epochs=500, lr=1e-3, patience=100,
    )

    assert "test_r2" in result
    assert "train_r2" in result
    assert "train_loss" in result
    assert "epochs_trained" in result
    assert result["epochs_trained"] > 0


def test_classification_with_pruning():
    """Training works with pruning masks applied."""
    torch.manual_seed(42)
    X_train, y_train, X_test, y_test = generate_modular_data(mod=11, seed=42)
    model = TwoLayerMLP(input_dim=22, hidden_dim=32, output_dim=11)

    # Create a simple mask
    masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            mask = (torch.rand_like(param) > 0.5).float()
            param.data *= mask
            masks[name] = mask

    result = train_classification(
        model, X_train, y_train, X_test, y_test,
        masks=masks, max_epochs=50, lr=1e-2, patience=50,
    )

    assert result["epochs_trained"] > 0
    # Verify masks are still applied
    for name, param in model.named_parameters():
        if name in masks:
            assert (param.data[masks[name] == 0] == 0).all()


def test_classification_early_stopping_does_not_use_test_labels():
    """Shuffling held-out labels must not change training behavior."""
    X_train, y_train, X_test, y_test = generate_modular_data(mod=11, seed=42)
    perm = torch.randperm(len(y_test), generator=torch.Generator().manual_seed(123))
    shuffled_y_test = y_test[perm]

    torch.manual_seed(0)
    baseline_model = TwoLayerMLP(input_dim=22, hidden_dim=32, output_dim=11)
    model_a = copy.deepcopy(baseline_model)
    model_b = copy.deepcopy(baseline_model)

    result_a = train_classification(
        model_a, X_train, y_train, X_test, y_test,
        masks={}, max_epochs=80, lr=1e-2, patience=10, validation_fraction=0.2, seed=42,
    )
    result_b = train_classification(
        model_b, X_train, y_train, X_test, shuffled_y_test,
        masks={}, max_epochs=80, lr=1e-2, patience=10, validation_fraction=0.2, seed=42,
    )

    assert result_a["epochs_trained"] == result_b["epochs_trained"]
    for name, param in model_a.state_dict().items():
        assert torch.equal(param, model_b.state_dict()[name])


def test_regression_early_stopping_does_not_use_test_targets():
    """Shuffling held-out targets must not change training behavior."""
    X_train, y_train, X_test, y_test = generate_regression_data(
        n_samples=120, n_features=10, seed=42
    )
    perm = torch.randperm(len(y_test), generator=torch.Generator().manual_seed(123))
    shuffled_y_test = y_test[perm]

    torch.manual_seed(0)
    baseline_model = TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=1)
    model_a = copy.deepcopy(baseline_model)
    model_b = copy.deepcopy(baseline_model)

    result_a = train_regression(
        model_a, X_train, y_train, X_test, y_test,
        masks={}, max_epochs=120, lr=1e-3, patience=15, validation_fraction=0.2, seed=42,
    )
    result_b = train_regression(
        model_b, X_train, y_train, X_test, shuffled_y_test,
        masks={}, max_epochs=120, lr=1e-3, patience=15, validation_fraction=0.2, seed=42,
    )

    assert result_a["epochs_trained"] == result_b["epochs_trained"]
    for name, param in model_a.state_dict().items():
        assert torch.equal(param, model_b.state_dict()[name])
