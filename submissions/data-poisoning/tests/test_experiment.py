"""Tests for experiment runner."""

from src.experiment import ExperimentConfig, run_single, run_sweep
from src.data import generate_gaussian_clusters


class TestRunSingle:
    """Tests for run_single."""

    def test_returns_result(self):
        config = ExperimentConfig(n_samples=50, n_epochs=10)
        X, y = generate_gaussian_clusters(n_samples=50, seed=42)
        result = run_single(config, poison_fraction=0.0, hidden_width=32, seed=42, X=X, y_clean=y)
        assert 0.0 <= result.test_accuracy <= 1.0
        assert 0.0 <= result.train_accuracy <= 1.0
        assert result.poison_fraction == 0.0
        assert result.hidden_width == 32

    def test_poisoned_lower_accuracy(self):
        config = ExperimentConfig(n_samples=100, n_epochs=50)
        X, y = generate_gaussian_clusters(n_samples=100, seed=42)

        clean_result = run_single(config, 0.0, 64, 42, X, y)
        poison_result = run_single(config, 0.5, 64, 42, X, y)

        # Heavy poisoning should degrade test accuracy
        assert poison_result.test_accuracy < clean_result.test_accuracy + 0.1


class TestRunSweep:
    """Tests for run_sweep (mini version)."""

    def test_mini_sweep(self):
        config = ExperimentConfig(
            n_samples=50,
            n_epochs=5,
            poison_fractions=(0.0, 0.5),
            hidden_widths=(32,),
            seeds=(42,),
        )
        results = run_sweep(config)
        assert len(results) == 2  # 2 fractions x 1 width x 1 seed
