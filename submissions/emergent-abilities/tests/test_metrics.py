"""Tests for src/metrics.py -- scoring functions and curve fitting."""

import math
import numpy as np
from src.metrics import (
    exact_match_from_token_accuracy,
    partial_credit_from_token_accuracy,
    token_edit_distance,
    brier_score,
    sigmoid_fit,
    linear_fit,
    compute_aic,
    compute_bic,
)


# ── Exact match ──────────────────────────────────────────────────────────────

def test_exact_match_perfect():
    """Perfect per-token accuracy -> perfect exact match."""
    assert exact_match_from_token_accuracy(1.0, 4) == 1.0


def test_exact_match_zero():
    """Zero per-token accuracy -> zero exact match."""
    assert exact_match_from_token_accuracy(0.0, 4) == 0.0


def test_exact_match_partial():
    """Partial per-token accuracy -> p^n."""
    result = exact_match_from_token_accuracy(0.9, 4)
    expected = 0.9 ** 4
    assert abs(result - expected) < 1e-10


def test_exact_match_single_token():
    """With 1 token, exact match equals per-token accuracy."""
    assert exact_match_from_token_accuracy(0.7, 1) == 0.7


def test_exact_match_many_tokens():
    """More tokens -> lower exact match for same per-token accuracy."""
    em_short = exact_match_from_token_accuracy(0.8, 2)
    em_long = exact_match_from_token_accuracy(0.8, 8)
    assert em_long < em_short


# ── Partial credit ───────────────────────────────────────────────────────────

def test_partial_credit_equals_p():
    """Partial credit equals per-token accuracy regardless of n."""
    assert partial_credit_from_token_accuracy(0.7, 4) == 0.7
    assert partial_credit_from_token_accuracy(0.7, 1) == 0.7
    assert partial_credit_from_token_accuracy(0.7, 100) == 0.7


# ── Token edit distance ─────────────────────────────────────────────────────

def test_token_edit_distance_formula():
    """Token edit distance = n * (1 - p)."""
    result = token_edit_distance(0.8, 5)
    expected = 5 * (1 - 0.8)
    assert abs(result - expected) < 1e-10


def test_token_edit_distance_perfect():
    """Perfect accuracy -> zero edit distance."""
    assert token_edit_distance(1.0, 5) == 0.0


def test_token_edit_distance_zero():
    """Zero accuracy -> max edit distance."""
    assert token_edit_distance(0.0, 5) == 5.0


# ── Brier score ──────────────────────────────────────────────────────────────

def test_brier_score_perfect():
    """Perfect prediction -> Brier score = 0."""
    assert brier_score(1.0, 1) == 0.0
    assert brier_score(0.0, 0) == 0.0


def test_brier_score_worst():
    """Worst prediction -> Brier score = 1."""
    assert brier_score(0.0, 1) == 1.0
    assert brier_score(1.0, 0) == 1.0


def test_brier_score_middle():
    """50% prediction -> Brier score = 0.25."""
    assert abs(brier_score(0.5, 1) - 0.25) < 1e-10


# ── Sigmoid fit ──────────────────────────────────────────────────────────────

def test_sigmoid_fit_recovers_known():
    """Sigmoid fit recovers parameters of known sigmoid data."""
    np.random.seed(42)
    x = np.linspace(-5, 5, 50)
    # True sigmoid: L=1, k=1, x0=0
    y_true = 1.0 / (1.0 + np.exp(-x))
    y = y_true + np.random.normal(0, 0.02, len(x))
    y = np.clip(y, 0, 1)

    params, r_squared, residuals = sigmoid_fit(x, y)
    assert r_squared > 0.95, f"R^2 too low: {r_squared}"
    # x0 should be near 0
    assert abs(params["x0"]) < 1.0, f"x0 too far from 0: {params['x0']}"


def test_sigmoid_fit_returns_r_squared():
    """Sigmoid fit returns R-squared between 0 and 1."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([0.1, 0.15, 0.3, 0.7, 0.9])
    _, r_squared, _ = sigmoid_fit(x, y)
    assert 0.0 <= r_squared <= 1.0


# ── Linear fit ───────────────────────────────────────────────────────────────

def test_linear_fit_recovers_known():
    """Linear fit recovers parameters of known linear data."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = 2.0 * x + 1.0  # slope=2, intercept=1
    params, r_squared, _ = linear_fit(x, y)
    assert abs(params["slope"] - 2.0) < 1e-6
    assert abs(params["intercept"] - 1.0) < 1e-6
    assert abs(r_squared - 1.0) < 1e-6


def test_linear_fit_returns_r_squared():
    """Linear fit returns R-squared between 0 and 1."""
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    _, r_squared, _ = linear_fit(x, y)
    assert 0.0 <= r_squared <= 1.0


# ── AIC / BIC ───────────────────────────────────────────────────────────────

def test_aic_known_value():
    """AIC = n * ln(RSS/n) + 2k for known values."""
    n, rss, k = 10, 1.0, 2
    aic = compute_aic(n, rss, k)
    expected = n * math.log(rss / n) + 2 * k
    assert abs(aic - expected) < 1e-10


def test_bic_known_value():
    """BIC = n * ln(RSS/n) + k * ln(n) for known values."""
    n, rss, k = 10, 1.0, 2
    bic = compute_bic(n, rss, k)
    expected = n * math.log(rss / n) + k * math.log(n)
    assert abs(bic - expected) < 1e-10


def test_aic_penalizes_more_params():
    """More parameters -> higher AIC (more penalty)."""
    n, rss = 10, 1.0
    aic_simple = compute_aic(n, rss, k=2)
    aic_complex = compute_aic(n, rss, k=4)
    assert aic_complex > aic_simple
