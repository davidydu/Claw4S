"""Tests for MLP model and training."""

import torch
import numpy as np
from torch.utils.data import TensorDataset

from src.model import MLP, evaluate_accuracy, train_model


class TestMLP:
    """Tests for the MLP architecture."""

    def test_output_shape(self):
        model = MLP(n_features=10, hidden_width=32, n_classes=5)
        x = torch.randn(16, 10)
        out = model(x)
        assert out.shape == (16, 5)

    def test_different_widths(self):
        for width in [32, 64, 128]:
            model = MLP(n_features=10, hidden_width=width, n_classes=5)
            n_params = sum(p.numel() for p in model.parameters())
            # width * 10 + width (first layer) + 5 * width + 5 (second layer)
            expected = width * 10 + width + 5 * width + 5
            assert n_params == expected, f"Width {width}: {n_params} != {expected}"

    def test_reproducibility(self):
        torch.manual_seed(42)
        m1 = MLP(10, 32, 5)
        torch.manual_seed(42)
        m2 = MLP(10, 32, 5)
        for p1, p2 in zip(m1.parameters(), m2.parameters()):
            torch.testing.assert_close(p1, p2)


class TestTrainModel:
    """Tests for model training."""

    def test_loss_decreases(self):
        torch.manual_seed(42)
        X = torch.randn(100, 10)
        y = torch.randint(0, 5, (100,))
        dataset = TensorDataset(X, y)
        model = MLP(10, 64, 5)
        losses = train_model(model, dataset, n_epochs=50, seed=42)
        assert losses[-1] < losses[0], "Loss should decrease during training"

    def test_overfits_small_data(self):
        torch.manual_seed(42)
        # Small dataset that should be easy to overfit
        X = torch.randn(20, 10)
        y = torch.zeros(20, dtype=torch.long)  # All same class
        dataset = TensorDataset(X, y)
        model = MLP(10, 128, 5)
        train_model(model, dataset, n_epochs=200, lr=0.05, seed=42)
        acc = evaluate_accuracy(model, X, y)
        assert acc > 0.9, f"Should overfit small data, got acc={acc:.3f}"


class TestEvaluateAccuracy:
    """Tests for accuracy evaluation."""

    def test_perfect_accuracy(self):
        model = MLP(10, 32, 5)
        X = torch.randn(50, 10)
        with torch.no_grad():
            logits = model(X)
            y = logits.argmax(dim=1)
        acc = evaluate_accuracy(model, X, y)
        assert abs(acc - 1.0) < 1e-6

    def test_random_baseline(self):
        torch.manual_seed(42)
        model = MLP(10, 32, 5)
        X = torch.randn(1000, 10)
        y = torch.randint(0, 5, (1000,))
        acc = evaluate_accuracy(model, X, y)
        # Random accuracy for 5 classes should be ~0.2
        assert 0.05 < acc < 0.6, f"Random accuracy should be near chance, got {acc:.3f}"
