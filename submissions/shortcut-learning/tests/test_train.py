"""Tests for training and evaluation."""

import torch
from torch.utils.data import TensorDataset

from src.train import train_model, evaluate_accuracy
from src.data import generate_dataset


def test_train_returns_eval_model():
    """Trained model should be in eval mode."""
    data = generate_dataset(n_train=100, n_test=50, seed=42)
    model = train_model(
        train_dataset=data["train_dataset"],
        input_dim=11,
        hidden_dim=32,
        weight_decay=0.0,
        seed=42,
        epochs=5,
    )
    assert not model.training, "Model should be in eval mode after training"


def test_evaluate_accuracy_range():
    """Accuracy should be between 0 and 1."""
    data = generate_dataset(n_train=200, n_test=100, seed=42)
    model = train_model(
        train_dataset=data["train_dataset"],
        input_dim=11,
        hidden_dim=32,
        seed=42,
        epochs=10,
    )
    acc = evaluate_accuracy(model, data["test_with_shortcut"])
    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of range"


def test_train_improves_over_random():
    """After training, model should do better than random (50%) on training data."""
    data = generate_dataset(n_train=500, n_test=100, seed=42)
    model = train_model(
        train_dataset=data["train_dataset"],
        input_dim=11,
        hidden_dim=64,
        seed=42,
        epochs=50,
    )
    train_acc = evaluate_accuracy(model, data["train_dataset"])
    assert train_acc > 0.6, \
        f"Expected training accuracy > 0.6 after 50 epochs, got {train_acc:.3f}"


def test_perfect_dataset():
    """Model should achieve ~100% on a trivially separable dataset."""
    # Create perfectly separable data
    torch.manual_seed(42)
    X = torch.cat([torch.ones(50, 5), -torch.ones(50, 5)], dim=0)
    y = torch.cat([torch.ones(50), torch.zeros(50)], dim=0).long()
    ds = TensorDataset(X, y)

    model = train_model(
        train_dataset=ds,
        input_dim=5,
        hidden_dim=32,
        seed=42,
        epochs=100,
    )
    acc = evaluate_accuracy(model, ds)
    assert acc > 0.95, f"Expected > 0.95 on trivial dataset, got {acc:.3f}"
