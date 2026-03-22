"""Tests for curve fitting functions."""

import numpy as np
from src.curve_fitting import (
    power_law,
    exponential,
    stretched_exponential,
    log_power,
    compute_aic,
    compute_bic,
    fit_single_curve,
    fit_all_forms,
    FUNCTIONAL_FORMS,
)


def test_power_law_shape():
    t = np.arange(1, 100, dtype=float)
    y = power_law(t, a=1.0, beta=0.5, L_inf=0.1)
    assert y.shape == t.shape
    # Should be decreasing
    assert y[0] > y[-1]


def test_exponential_shape():
    t = np.arange(1, 100, dtype=float)
    y = exponential(t, a=1.0, lam=0.05, L_inf=0.0)
    assert y.shape == t.shape
    assert y[0] > y[-1]


def test_stretched_exponential_shape():
    t = np.arange(1, 100, dtype=float)
    y = stretched_exponential(t, a=1.0, tau=50.0, gamma=0.5, L_inf=0.0)
    assert y.shape == t.shape
    assert y[0] > y[-1]


def test_log_power_shape():
    t = np.arange(2, 100, dtype=float)
    y = log_power(t, a=5.0, beta=1.0, L_inf=0.0)
    assert y.shape == t.shape
    assert y[0] > y[-1]


def test_aic_basic():
    # AIC = n * ln(RSS/n) + 2k
    aic = compute_aic(n=100, k=3, rss=10.0)
    expected = 100 * np.log(10.0 / 100) + 2 * 3
    assert abs(aic - expected) < 1e-10


def test_bic_basic():
    # BIC = n * ln(RSS/n) + k * ln(n)
    bic = compute_bic(n=100, k=3, rss=10.0)
    expected = 100 * np.log(10.0 / 100) + 3 * np.log(100)
    assert abs(bic - expected) < 1e-10


def test_aic_zero_rss():
    assert compute_aic(100, 3, 0.0) == np.inf


def test_fit_single_curve_power_law():
    # Generate synthetic power law data
    t = np.arange(10, 500, dtype=float)
    y = power_law(t, a=2.0, beta=0.5, L_inf=0.1)
    y += 0.001 * np.random.default_rng(42).standard_normal(len(t))

    result = fit_single_curve(t, y, "power_law")
    assert result["converged"]
    assert result["rss"] < 0.1
    assert abs(result["params"]["beta"] - 0.5) < 0.1


def test_fit_single_curve_exponential():
    t = np.arange(10, 500, dtype=float)
    y = exponential(t, a=2.0, lam=0.01, L_inf=0.05)
    y += 0.001 * np.random.default_rng(42).standard_normal(len(t))

    result = fit_single_curve(t, y, "exponential")
    assert result["converged"]
    assert result["rss"] < 0.1


def test_fit_all_forms_returns_sorted():
    # Synthetic data from a known form
    epochs = list(range(1, 301))
    losses = [2.0 * e ** (-0.5) + 0.1 for e in epochs]

    results = fit_all_forms(epochs, losses, skip_epochs=10)
    assert len(results) == 4
    # Should be sorted by AIC
    aics = [r["aic"] for r in results]
    assert aics == sorted(aics)


def test_functional_forms_registry():
    assert set(FUNCTIONAL_FORMS.keys()) == {
        "power_law",
        "exponential",
        "stretched_exp",
        "log_power",
    }
    for name, form in FUNCTIONAL_FORMS.items():
        assert "func" in form
        assert "p0" in form
        assert "bounds" in form
        assert "n_params" in form
        assert len(form["p0"]) == form["n_params"]
