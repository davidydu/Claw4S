# tests/test_train.py
"""Tests for the training loop."""

import torch
from src.model import MLP
from src.train import train_model, compute_accuracy


def test_compute_accuracy():
    """Accuracy computation on known data."""
    model = MLP(input_dim=2, hidden_dim=10, num_classes=2)
    X = torch.randn(10, 2)
    y = torch.zeros(10, dtype=torch.long)  # All same class

    acc = compute_accuracy(model, X, y)
    assert 0.0 <= acc <= 1.0


def test_train_model_returns_result():
    """Training returns a TrainResult with expected fields."""
    model = MLP(input_dim=5, hidden_dim=20, num_classes=3)
    X = torch.randn(20, 5)
    y = torch.randint(0, 3, (20,))

    result = train_model(model, X, y, max_epochs=50, log_interval=0)

    assert hasattr(result, "final_train_acc")
    assert hasattr(result, "final_train_loss")
    assert hasattr(result, "convergence_epoch")
    assert hasattr(result, "epochs_run")
    assert hasattr(result, "loss_history")
    assert len(result.loss_history) == result.epochs_run
    assert 0.0 <= result.final_train_acc <= 1.0
    assert result.final_train_loss >= 0.0


def test_large_model_memorizes():
    """A sufficiently large model should memorize small random data."""
    torch.manual_seed(42)
    X = torch.randn(10, 5)
    y = torch.randint(0, 3, (10,))

    model = MLP(input_dim=5, hidden_dim=100, num_classes=3)
    result = train_model(model, X, y, max_epochs=2000, log_interval=0, seed=42)

    assert result.final_train_acc >= 0.9, (
        f"Large model should memorize: got acc={result.final_train_acc:.4f}"
    )


def test_training_reduces_loss():
    """Training should reduce loss over time."""
    model = MLP(input_dim=5, hidden_dim=20, num_classes=3)
    X = torch.randn(20, 5)
    y = torch.randint(0, 3, (20,))

    result = train_model(model, X, y, max_epochs=100, log_interval=0)

    # Loss at end should be less than at start
    assert result.loss_history[-1] < result.loss_history[0], "Training should reduce loss"
