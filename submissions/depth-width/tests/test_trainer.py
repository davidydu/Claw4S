"""Tests for the training loop."""

import torch

from src.models import FlexibleMLP
from src.tasks import make_sparse_parity_data, make_regression_data
from src.trainer import train_model


class TestTrainer:
    """Tests for train_model function."""

    def test_classification_returns_metrics(self):
        """Training on classification task should return required metrics."""
        data = make_sparse_parity_data(
            n_bits=10, k_relevant=3, n_train=100, n_test=50, seed=42
        )
        model = FlexibleMLP(10, 32, 2, 2)
        results = train_model(
            model, data, max_epochs=10, lr=1e-3, seed=42, log_interval=100
        )
        assert "final_train_metric" in results
        assert "final_test_metric" in results
        assert "best_test_metric" in results
        assert "best_epoch" in results
        assert "total_epochs" in results
        assert "training_time_sec" in results
        assert results["metric_name"] == "accuracy"
        assert 0.0 <= results["best_test_metric"] <= 1.0

    def test_regression_returns_metrics(self):
        """Training on regression task should return required metrics."""
        data = make_regression_data(
            n_train=100, n_test=50, input_dim=4, seed=42
        )
        model = FlexibleMLP(4, 16, 1, 2)
        results = train_model(
            model, data, max_epochs=10, lr=1e-3, seed=42, log_interval=100
        )
        assert "final_train_metric" in results
        assert "final_test_metric" in results
        assert results["metric_name"] == "r_squared"
        assert "training_time_sec" in results

    def test_early_stopping(self):
        """Training should stop early if no improvement."""
        data = make_regression_data(
            n_train=50, n_test=20, input_dim=4, seed=42
        )
        model = FlexibleMLP(4, 8, 1, 1)
        results = train_model(
            model, data, max_epochs=5000, patience=10, lr=1e-3,
            seed=42, log_interval=10000,
        )
        assert results["total_epochs"] < 5000

    def test_loss_decreases(self):
        """Loss should generally decrease over training."""
        data = make_regression_data(
            n_train=200, n_test=50, input_dim=4, seed=42
        )
        model = FlexibleMLP(4, 32, 1, 2)
        results = train_model(
            model, data, max_epochs=100, lr=1e-3, seed=42, log_interval=10000
        )
        losses = results["epoch_losses"]
        # First loss should be higher than last loss
        assert losses[0] > losses[-1]
