"""Tests for synthetic data generation."""

import torch

from src.data import generate_regression_data, get_data_summary


class TestGenerateRegressionData:
    """Tests for generate_regression_data."""

    def test_output_shapes(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=100, n_test=50, d=10
        )
        assert X_train.shape == (100, 10)
        assert y_train.shape == (100, 1)
        assert X_test.shape == (50, 10)
        assert y_test.shape == (50, 1)

    def test_deterministic_with_seed(self):
        r1 = generate_regression_data(n_train=50, n_test=20, d=5, seed=42)
        r2 = generate_regression_data(n_train=50, n_test=20, d=5, seed=42)
        for a, b in zip(r1, r2):
            assert torch.allclose(a, b), "Same seed should give same data"

    def test_different_seeds_differ(self):
        r1 = generate_regression_data(n_train=50, n_test=20, d=5, seed=42)
        r2 = generate_regression_data(n_train=50, n_test=20, d=5, seed=99)
        assert not torch.allclose(r1[0], r2[0]), "Different seeds should differ"

    def test_noise_affects_targets(self):
        _, y_low, _, _ = generate_regression_data(
            n_train=200, n_test=50, d=5, noise_std=0.01, seed=42
        )
        _, y_high, _, _ = generate_regression_data(
            n_train=200, n_test=50, d=5, noise_std=10.0, seed=42
        )
        assert y_high.std() > y_low.std()

    def test_float32_dtype(self):
        X_train, y_train, X_test, y_test = generate_regression_data()
        assert X_train.dtype == torch.float32
        assert y_train.dtype == torch.float32

    def test_default_params(self):
        X_train, y_train, X_test, y_test = generate_regression_data()
        assert X_train.shape == (200, 20)
        assert X_test.shape == (200, 20)


class TestGetDataSummary:
    """Tests for get_data_summary."""

    def test_summary_keys(self):
        data = generate_regression_data(n_train=100, n_test=50, d=10)
        summary = get_data_summary(*data)
        assert summary["n_train"] == 100
        assert summary["n_test"] == 50
        assert summary["d"] == 10
        assert "y_train_mean" in summary
        assert "y_train_std" in summary
