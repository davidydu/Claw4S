"""Tests for phase transition analysis module."""
import numpy as np
from src.analysis import (
    fit_sigmoid, estimate_kc_susceptibility, fit_critical_exponent,
    fit_finite_size_scaling, bootstrap_ci, compute_statistics,
)


def test_fit_sigmoid():
    """Sigmoid fit should recover K_c from synthetic data."""
    K_vals = np.linspace(0, 3, 20)
    # Synthetic sigmoid: r = 1/(1+exp(-5*(K-1.0)))
    r_vals = 1 / (1 + np.exp(-5 * (K_vals - 1.0)))
    kc, params = fit_sigmoid(K_vals, r_vals)
    assert abs(kc - 1.0) < 0.1


def test_susceptibility_peak():
    """Susceptibility should peak near K_c."""
    K_vals = np.linspace(0, 3, 20)
    # r values with transition at K=1.0
    r_mean = 1 / (1 + np.exp(-5 * (K_vals - 1.0)))
    r_var = 0.01 * np.exp(-((K_vals - 1.0) ** 2) / 0.1)  # peaks at K=1
    kc = estimate_kc_susceptibility(K_vals, r_var, n=12)
    assert abs(kc - 1.0) < 0.3


def test_critical_exponent():
    """β should be recoverable from synthetic power-law data."""
    K_vals = np.array([1.1, 1.2, 1.5, 2.0, 2.5, 3.0])
    kc = 1.0
    beta_true = 0.5
    r_vals = (K_vals - kc) ** beta_true
    beta, r_squared = fit_critical_exponent(K_vals, r_vals, kc)
    assert abs(beta - 0.5) < 0.1
    assert r_squared > 0.95


def test_compute_statistics_groups():
    """Should group records and compute mean/std."""
    records = [
        {"topology": "all-to-all", "n": 12, "sigma": 0.5, "K": 1.0, "seed": i,
         "final_r": 0.7 + i * 0.01,
         "evaluator_scores": {"kuramoto_order": 0.7, "spatial_alignment": 0.6,
                              "velocity_synchrony": 0.65, "pairwise_entrainment": 0.7}}
        for i in range(5)
    ]
    stats = compute_statistics(records)
    assert len(stats) == 1
    assert stats[0]["n_seeds"] == 5
    assert abs(stats[0]["mean_r"] - 0.72) < 0.01


def test_finite_size_scaling():
    """Should extrapolate K_c(inf) from K_c(N) data."""
    kc_by_n = {6: 1.1, 12: 0.95, 24: 0.85}
    kc_inf, nu = fit_finite_size_scaling(kc_by_n)
    assert 0.5 < kc_inf < 1.5
    assert nu > 0


def test_bootstrap_ci():
    """Should return a confidence interval tuple (low, high)."""
    values = [0.8, 0.82, 0.79, 0.81, 0.83]
    low, high = bootstrap_ci(values, confidence=0.95, n_bootstrap=1000, seed=42)
    assert low < high
    assert low > 0.7
    assert high < 0.9
