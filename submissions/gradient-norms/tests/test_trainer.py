"""Tests for the training loop and metric tracking."""

from src.data import make_modular_addition_dataset, make_regression_dataset
from src.trainer import train_and_track


class TestTrainer:
    def test_classification_run(self):
        """Short training run on modular addition."""
        dataset = make_modular_addition_dataset(modulus=11, frac=0.7, seed=42)
        result = train_and_track(
            dataset, hidden_dim=16, n_epochs=50, seed=42
        )
        assert result["metric_name"] == "accuracy"
        assert result["task_name"] == "modular_addition"
        assert len(result["epochs"]) == 50
        assert len(result["train_loss"]) == 50
        assert len(result["test_loss"]) == 50
        assert len(result["train_metric"]) == 50
        assert len(result["test_metric"]) == 50
        assert "layer1" in result["grad_norms"]
        assert "layer2" in result["grad_norms"]
        assert len(result["grad_norms"]["layer1"]) == 50

    def test_regression_run(self):
        """Short training run on regression."""
        dataset = make_regression_dataset(n_samples=100, frac=0.7, seed=42)
        result = train_and_track(
            dataset, hidden_dim=16, n_epochs=50, seed=42
        )
        assert result["metric_name"] == "r_squared"
        assert result["task_name"] == "regression"
        assert len(result["epochs"]) == 50

    def test_grad_norms_positive(self):
        """Gradient norms should be positive."""
        dataset = make_modular_addition_dataset(modulus=11, frac=0.7, seed=42)
        result = train_and_track(
            dataset, hidden_dim=16, n_epochs=20, seed=42
        )
        for name, norms in result["grad_norms"].items():
            assert all(n >= 0 for n in norms), f"{name} has negative gradient norms"

    def test_weight_norms_positive(self):
        """Weight norms should be positive."""
        dataset = make_modular_addition_dataset(modulus=11, frac=0.7, seed=42)
        result = train_and_track(
            dataset, hidden_dim=16, n_epochs=20, seed=42
        )
        for name, norms in result["weight_norms"].items():
            assert all(n > 0 for n in norms), f"{name} has non-positive weight norms"

    def test_reproducibility(self):
        """Same seed should give same results."""
        dataset = make_modular_addition_dataset(modulus=11, frac=0.7, seed=42)
        r1 = train_and_track(dataset, hidden_dim=16, n_epochs=20, seed=42)
        r2 = train_and_track(dataset, hidden_dim=16, n_epochs=20, seed=42)
        assert r1["train_loss"] == r2["train_loss"]
        assert r1["grad_norms"]["layer1"] == r2["grad_norms"]["layer1"]
