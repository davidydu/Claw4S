"""Tests for src/trainer.py."""

import torch
from src.models import create_model
from src.data import make_regression_data
from src.trainer import train_with_tracking


def test_training_produces_history():
    """Training returns history with expected keys and lengths."""
    data = make_regression_data(n_train=100, n_test=50, input_dim=5, seed=42)
    model = create_model(
        input_dim=data["input_dim"],
        hidden_dim=16,
        output_dim=data["output_dim"],
        seed=42,
    )
    history = train_with_tracking(
        model=model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        task_type="regression",
        n_epochs=100,
        track_every=25,
        seed=42,
    )

    expected_keys = [
        "epochs", "train_loss", "test_loss", "train_acc", "test_acc",
        "dead_neuron_fraction", "near_dead_fraction", "zero_fraction",
        "activation_entropy", "mean_activation_magnitude",
    ]
    for key in expected_keys:
        assert key in history, f"Missing key: {key}"

    # epochs 0, 25, 50, 75, 100 -> 5 tracking points
    assert len(history["epochs"]) == 5
    assert history["epochs"][0] == 0
    assert history["epochs"][-1] == 100


def test_training_loss_decreases():
    """Training loss should decrease over epochs."""
    data = make_regression_data(n_train=200, n_test=50, input_dim=5, seed=42)
    model = create_model(
        input_dim=data["input_dim"],
        hidden_dim=32,
        output_dim=data["output_dim"],
        seed=42,
    )
    history = train_with_tracking(
        model=model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        task_type="regression",
        n_epochs=200,
        track_every=50,
        seed=42,
    )

    # Loss at end should be less than at start
    assert history["train_loss"][-1] < history["train_loss"][0]


def test_dead_fraction_valid_range():
    """Dead neuron fraction stays in [0, 1] throughout training."""
    data = make_regression_data(n_train=100, n_test=50, input_dim=5, seed=42)
    model = create_model(
        input_dim=data["input_dim"],
        hidden_dim=16,
        output_dim=data["output_dim"],
        seed=42,
    )
    history = train_with_tracking(
        model=model,
        x_train=data["x_train"],
        y_train=data["y_train"],
        x_test=data["x_test"],
        y_test=data["y_test"],
        task_type="regression",
        n_epochs=50,
        track_every=25,
        seed=42,
    )

    for df in history["dead_neuron_fraction"]:
        assert 0.0 <= df <= 1.0, f"Dead fraction {df} out of range"

    for zf in history["zero_fraction"]:
        assert 0.0 <= zf <= 1.0, f"Zero fraction {zf} out of range"
