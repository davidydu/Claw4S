"""Tests for scaling law curve fitting."""

import numpy as np
import pytest

from src.scaling import power_law, fit_scaling_law, bootstrap_alpha_ci


class TestPowerLaw:
    """Tests for the power law function."""

    def test_known_values(self):
        # L(100) = 2 * 100^(-0.5) + 1 = 2 * 0.1 + 1 = 1.2
        result = power_law(np.array([100.0]), 2.0, 0.5, 1.0)
        np.testing.assert_allclose(result, [1.2], rtol=1e-6)

    def test_monotone_decreasing(self):
        n = np.array([10, 100, 1000, 10000], dtype=float)
        losses = power_law(n, 5.0, 0.3, 0.5)
        # Should be decreasing
        assert all(losses[i] > losses[i + 1] for i in range(len(losses) - 1))

    def test_approaches_l_inf(self):
        n = np.array([1e10], dtype=float)
        loss = power_law(n, 5.0, 0.3, 0.5)
        np.testing.assert_allclose(loss, [0.5], atol=0.01)


class TestFitScalingLaw:
    """Tests for scaling law fitting."""

    def test_recovers_known_parameters(self):
        """Fit should recover parameters from noise-free synthetic data."""
        n = np.array([100, 200, 500, 1000, 2000], dtype=float)
        true_a, true_alpha, true_l_inf = 10.0, 0.4, 0.5
        losses = power_law(n, true_a, true_alpha, true_l_inf)

        fit = fit_scaling_law(n, losses)
        np.testing.assert_allclose(fit["a"], true_a, rtol=0.1)
        np.testing.assert_allclose(fit["alpha"], true_alpha, rtol=0.1)
        np.testing.assert_allclose(fit["l_inf"], true_l_inf, rtol=0.1)
        assert fit["r_squared"] > 0.99

    def test_noisy_data_reasonable_fit(self):
        """Fit should still work with modest noise."""
        rng = np.random.RandomState(42)
        n = np.array([50, 100, 200, 500, 1000], dtype=float)
        losses = power_law(n, 5.0, 0.3, 1.0) + rng.normal(0, 0.05, size=5)

        fit = fit_scaling_law(n, losses)
        assert fit["alpha"] > 0
        assert fit["r_squared"] > 0.7

    def test_output_keys(self):
        n = np.array([100, 200, 500, 1000, 2000], dtype=float)
        losses = power_law(n, 5.0, 0.3, 1.0)
        fit = fit_scaling_law(n, losses)
        assert set(fit.keys()) == {"a", "alpha", "l_inf", "r_squared", "residuals"}

    def test_explicitly_uses_bounded_trf_solver(self, monkeypatch):
        """Bounded fits should explicitly request scipy's trust-region solver."""
        recorded = {}

        def fake_curve_fit(*args, **kwargs):
            recorded["method"] = kwargs.get("method")
            return np.array([5.0, 0.3, 1.0]), np.eye(3)

        monkeypatch.setattr("src.scaling.curve_fit", fake_curve_fit)

        n = np.array([100, 200, 500], dtype=float)
        losses = power_law(n, 5.0, 0.3, 1.0)

        fit = fit_scaling_law(n, losses)

        assert recorded["method"] == "trf"
        assert fit["alpha"] == pytest.approx(0.3)


class TestBootstrapAlphaCI:
    """Tests for bootstrap confidence intervals of scaling exponents."""

    def test_returns_well_formed_interval(self):
        param_counts = np.array([100, 200, 400, 800, 1600], dtype=float)
        base = power_law(param_counts, 8.0, 0.35, 0.2)

        # losses_by_size has shape (n_sizes, n_seeds)
        losses_by_size = np.stack(
            [
                base * 1.03,
                base * 1.01,
                base * 0.99,
                base * 0.98,
            ],
            axis=1,
        )

        ci = bootstrap_alpha_ci(
            param_counts=param_counts,
            losses_by_size=losses_by_size,
            n_bootstrap=200,
            seed=42,
        )

        assert set(ci.keys()) == {
            "alpha_mean",
            "alpha_std",
            "alpha_ci95_low",
            "alpha_ci95_high",
            "n_bootstrap",
        }
        assert ci["n_bootstrap"] == 200
        assert ci["alpha_std"] >= 0
        assert ci["alpha_ci95_low"] <= ci["alpha_mean"] <= ci["alpha_ci95_high"]
        assert ci["alpha_ci95_low"] > 0
