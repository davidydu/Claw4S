"""Tests for sweep experiments."""

import src.sweep as sweep_module
from src.sweep import random_features_sweep, mlp_sweep, epoch_wise_sweep, run_all_sweeps


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


class TestRunAllSweepsVarianceNoise:
    def test_defaults_variance_noise_to_highest_noise(self, monkeypatch):
        captured_noise = []

        def fake_random_features_sweep(
            widths, n_train=200, n_test=200, d=20, noise_std=1.0, seed=42
        ):
            captured_noise.append(noise_std)
            width = widths[0]
            return [{
                "width": width,
                "n_params": width,
                "param_ratio": width / n_train,
                "train_loss": 0.0,
                "test_loss": 1.0 + noise_std,
            }]

        def fake_mlp_sweep(*args, **kwargs):
            return [{
                "width": 2,
                "n_params": 45,
                "param_ratio": 0.225,
                "train_loss": 0.5,
                "test_loss": 0.8,
            }]

        def fake_epoch_wise_sweep(*args, **kwargs):
            return {
                "width": 9,
                "n_params": 199,
                "param_ratio": 0.995,
                "epochs": [1],
                "train_losses": [1.0],
                "test_losses": [1.2],
            }

        monkeypatch.setattr(sweep_module, "random_features_sweep", fake_random_features_sweep)
        monkeypatch.setattr(sweep_module, "mlp_sweep", fake_mlp_sweep)
        monkeypatch.setattr(sweep_module, "epoch_wise_sweep", fake_epoch_wise_sweep)

        result = run_all_sweeps({
            "noise_levels": [0.2, 0.8],
            "rf_widths": [10],
            "mlp_widths": [2],
            "variance_seeds": [11, 22],
        })

        # First two calls are main RF sweeps, last two are variance sweeps.
        assert captured_noise[-2:] == [0.8, 0.8]
        assert result["metadata"]["variance_noise_std"] == 0.8
