"""Tests for training and evaluation."""

import torch
import numpy as np

from src.models import MLP
from src.data import build_datasets
from src.train import train_model, evaluate


class TestTrainModel:
    def test_loss_decreases(self):
        """Training loss should decrease over epochs on clean data."""
        train_ds, _, _ = build_datasets(
            n_samples=100, noise_frac=0.0, seed=42
        )
        model = MLP(in_features=10, hidden_width=32, n_hidden_layers=1, n_classes=5)
        losses = train_model(model, train_ds, n_epochs=50, seed=42)
        assert losses[-1] < losses[0], "Loss should decrease over training"

    def test_deterministic(self):
        """Same seed should give same losses."""
        train_ds, _, _ = build_datasets(n_samples=100, seed=42)

        model1 = MLP(in_features=10, hidden_width=32, n_hidden_layers=1, n_classes=5)
        torch.manual_seed(42)
        losses1 = train_model(model1, train_ds, n_epochs=20, seed=42)

        model2 = MLP(in_features=10, hidden_width=32, n_hidden_layers=1, n_classes=5)
        torch.manual_seed(42)
        losses2 = train_model(model2, train_ds, n_epochs=20, seed=42)

        np.testing.assert_allclose(losses1, losses2, rtol=1e-5)


class TestEvaluate:
    def test_accuracy_range(self):
        """Accuracy should be in [0, 1]."""
        train_ds, test_ds, _ = build_datasets(n_samples=100, seed=42)
        model = MLP(in_features=10, hidden_width=32, n_hidden_layers=1, n_classes=5)
        train_model(model, train_ds, n_epochs=10, seed=42)
        acc = evaluate(model, test_ds)
        assert 0.0 <= acc <= 1.0

    def test_trained_beats_random(self):
        """A trained model should beat chance (1/5 = 0.20)."""
        train_ds, test_ds, _ = build_datasets(
            n_samples=200, noise_frac=0.0, seed=42
        )
        model = MLP(in_features=10, hidden_width=64, n_hidden_layers=2, n_classes=5)
        train_model(model, train_ds, n_epochs=100, seed=42)
        acc = evaluate(model, test_ds)
        assert acc > 0.30, f"Trained model acc={acc:.3f}, expected > 0.30"
