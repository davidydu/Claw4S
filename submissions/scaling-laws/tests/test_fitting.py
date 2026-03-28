# tests/test_fitting.py
import numpy as np
from src.fitting import fit_scaling_law, parametric_bootstrap, FitResult


def test_fit_recovers_known_kaplan_params():
    """Fitting should recover parameters used to generate synthetic data."""
    np.random.seed(42)
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    true_a, true_alpha, true_linf = 5.0, 0.07, 1.5
    y = true_a * np.power(n, -true_alpha) + true_linf
    y += np.random.normal(0, 0.005, len(y))

    result = fit_scaling_law("kaplan", n, y)
    assert isinstance(result, FitResult)
    assert abs(result.params["alpha"] - true_alpha) < 0.02
    assert abs(result.params["l_inf"] - true_linf) < 0.1
    assert result.adj_r_squared > 0.95


def test_fit_result_has_required_fields():
    """FitResult should contain params, residuals, adj_r_squared, aic, bic."""
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    y = 5.0 * np.power(n, -0.07) + 1.5

    result = fit_scaling_law("kaplan", n, y)
    assert hasattr(result, "params")
    assert hasattr(result, "residuals")
    assert hasattr(result, "adj_r_squared")
    assert hasattr(result, "aic")
    assert hasattr(result, "bic")
    assert hasattr(result, "converged")
    assert result.converged is True


def test_bootstrap_returns_ci():
    """Parametric bootstrap should return confidence intervals for each parameter."""
    np.random.seed(42)
    n = np.array([1e8, 3e8, 1e9, 3e9, 1e10, 3e10, 1e11])
    y = 5.0 * np.power(n, -0.07) + 1.5 + np.random.normal(0, 0.01, 7)

    result = fit_scaling_law("kaplan", n, y)
    ci = parametric_bootstrap("kaplan", n, result, n_bootstrap=100, seed=42)
    assert "alpha" in ci
    assert len(ci["alpha"]) == 2  # (lower, upper)
    assert ci["alpha"][0] < ci["alpha"][1]
    assert "convergence_rate" in ci


def test_fit_handles_convergence_failure_gracefully():
    """Fitting with nonsensical data should return converged=False, not raise."""
    n = np.array([1.0, 2.0, 3.0])
    y = np.array([100.0, 100.0, 100.0])
    result = fit_scaling_law("kaplan", n, y)
    assert isinstance(result, FitResult)
