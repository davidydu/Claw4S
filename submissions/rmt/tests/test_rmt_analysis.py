"""Tests for RMT analysis functions."""

import numpy as np
from src.rmt_analysis import (
    compute_mp_bounds,
    marchenko_pastur_pdf,
    analyze_weight_matrix,
)


def test_mp_bounds_square():
    """For gamma=1 (square matrix), bounds are sigma^2*(1 +/- 1)^2."""
    lam_min, lam_max = compute_mp_bounds(gamma=1.0, sigma_sq=1.0)
    assert abs(lam_min - 0.0) < 1e-10  # (1-1)^2 = 0
    assert abs(lam_max - 4.0) < 1e-10  # (1+1)^2 = 4


def test_mp_bounds_rectangular():
    """For gamma=0.5, sigma^2=1: lambda_- = (1-sqrt(0.5))^2."""
    lam_min, lam_max = compute_mp_bounds(gamma=0.5, sigma_sq=1.0)
    expected_min = (1.0 - np.sqrt(0.5)) ** 2
    expected_max = (1.0 + np.sqrt(0.5)) ** 2
    assert abs(lam_min - expected_min) < 1e-10
    assert abs(lam_max - expected_max) < 1e-10


def test_mp_pdf_integrates_to_one():
    """MP PDF should integrate to approximately 1."""
    gamma = 0.5
    sigma_sq = 1.0
    lam_min, lam_max = compute_mp_bounds(gamma, sigma_sq)
    x = np.linspace(lam_min + 1e-6, lam_max - 1e-6, 5000)
    pdf = marchenko_pastur_pdf(x, gamma, sigma_sq)
    integral = np.trapezoid(pdf, x)
    assert abs(integral - 1.0) < 0.02  # Allow 2% numerical error


def test_mp_pdf_zero_outside_support():
    """PDF should be zero outside [lambda_-, lambda_+]."""
    gamma = 0.5
    sigma_sq = 1.0
    lam_min, lam_max = compute_mp_bounds(gamma, sigma_sq)
    x_outside = np.array([lam_min - 1.0, lam_max + 1.0, -1.0])
    pdf = marchenko_pastur_pdf(x_outside, gamma, sigma_sq)
    assert np.all(pdf == 0.0)


def test_random_matrix_low_ks():
    """A random Gaussian matrix should have low KS statistic."""
    rng = np.random.RandomState(42)
    M, N = 200, 50
    W = rng.randn(M, N) * 0.1  # sigma = 0.1
    result = analyze_weight_matrix(W, layer_name="random")
    # Random matrix should approximately match MP
    assert result["ks_statistic"] < 0.3
    assert result["outlier_fraction"] < 0.15


def test_structured_matrix_high_ks():
    """A matrix with planted structure should deviate from MP."""
    rng = np.random.RandomState(42)
    M, N = 200, 50
    W = rng.randn(M, N) * 0.1

    # Plant a strong rank-1 signal
    u = rng.randn(M, 1)
    v = rng.randn(1, N)
    W = W + 5.0 * (u @ v)  # Large signal spike

    result = analyze_weight_matrix(W, layer_name="structured")
    # Should have higher deviation
    assert result["spectral_norm_ratio"] > 2.0
    assert result["outlier_fraction"] > 0.0


def test_analyze_returns_all_fields():
    """Verify all expected fields are in the result dict."""
    rng = np.random.RandomState(42)
    W = rng.randn(50, 30)
    result = analyze_weight_matrix(W, layer_name="test")

    expected_fields = [
        "layer_name", "shape", "gamma", "sigma_sq",
        "lambda_minus", "lambda_plus", "max_eigenvalue", "min_eigenvalue",
        "ks_statistic", "ks_pvalue", "outlier_fraction", "n_outliers",
        "spectral_norm_ratio", "kl_divergence", "n_eigenvalues", "eigenvalues",
    ]
    for field in expected_fields:
        assert field in result, f"Missing field: {field}"
