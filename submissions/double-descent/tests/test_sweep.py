"""Tests for sweep experiments."""

from src.sweep import random_features_sweep, mlp_sweep, epoch_wise_sweep


class TestRandomFeaturesSweep:
    """Tests for random_features_sweep."""

    def test_output_structure(self):
        results = random_features_sweep(
            widths=[10, 50, 100],
            n_train=50, n_test=20, d=5,
            noise_std=0.5, seed=42,
        )
        assert len(results) == 3
        for r in results:
            assert "width" in r
            assert "n_params" in r
            assert "param_ratio" in r
            assert "train_loss" in r
            assert "test_loss" in r

    def test_widths_match(self):
        widths = [10, 50, 200]
        results = random_features_sweep(
            widths=widths,
            n_train=50, n_test=20, d=5,
            noise_std=0.5, seed=42,
        )
        assert [r["width"] for r in results] == widths

    def test_overparameterized_low_train_loss(self):
        results = random_features_sweep(
            widths=[200],
            n_train=50, n_test=20, d=5,
            noise_std=0.5, seed=42,
        )
        assert results[0]["train_loss"] < 0.01


class TestMlpSweep:
    """Tests for mlp_sweep."""

    def test_output_structure(self):
        results = mlp_sweep(
            widths=[5, 10, 20],
            n_train=50, n_test=20, d=5,
            noise_std=0.5, epochs=100, seed=42,
        )
        assert len(results) == 3
        for r in results:
            assert "width" in r
            assert "n_params" in r
            assert "train_loss" in r
            assert "test_loss" in r

    def test_params_increase_with_width(self):
        results = mlp_sweep(
            widths=[5, 10, 20, 50],
            n_train=50, n_test=20, d=5,
            noise_std=0.5, epochs=100, seed=42,
        )
        params = [r["n_params"] for r in results]
        assert params == sorted(params)


class TestEpochWiseSweep:
    """Tests for epoch_wise_sweep."""

    def test_output_structure(self):
        result = epoch_wise_sweep(
            width=10,
            n_train=50, n_test=20, d=5,
            noise_std=0.5, max_epochs=200,
            record_every=50, seed=42,
        )
        assert "width" in result
        assert "n_params" in result
        assert "epochs" in result
        assert "train_losses" in result
        assert "test_losses" in result
        assert len(result["epochs"]) == len(result["train_losses"])

    def test_epochs_recorded(self):
        result = epoch_wise_sweep(
            width=10,
            n_train=50, n_test=20, d=5,
            noise_std=0.5, max_epochs=200,
            record_every=50, seed=42,
        )
        # Should record at: 1, 50, 100, 150, 200 = 5 points
        assert len(result["epochs"]) == 5
