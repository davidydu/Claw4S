"""Tests for Benford's Law analysis module."""

import math

import numpy as np

from src.benford_analysis import (
    benford_expected,
    chi_squared_test,
    classify_mad,
    compute_digit_distribution,
    extract_leading_digits,
    generate_control_weights,
    kl_divergence,
    mad_from_benford,
)


def test_benford_expected_sums_to_one():
    """Benford expected probabilities sum to 1."""
    expected = benford_expected()
    assert len(expected) == 9
    total = sum(expected.values())
    assert abs(total - 1.0) < 1e-10


def test_benford_expected_values():
    """Benford digit 1 should be ~30.1%."""
    expected = benford_expected()
    assert abs(expected[1] - 0.30103) < 1e-4
    assert abs(expected[9] - math.log10(1 + 1 / 9)) < 1e-10
    # Monotonically decreasing
    for d in range(1, 9):
        assert expected[d] > expected[d + 1]


def test_extract_leading_digits_simple():
    """Leading digit extraction on known values."""
    weights = np.array([1.5, 2.3, 0.045, 0.00067, 9.1, 3.14])
    digits = extract_leading_digits(weights)
    expected = np.array([1, 2, 4, 6, 9, 3])
    np.testing.assert_array_equal(digits, expected)


def test_extract_leading_digits_negatives():
    """Negative values treated as absolute values."""
    weights = np.array([-1.5, -2.3, -0.045])
    digits = extract_leading_digits(weights)
    expected = np.array([1, 2, 4])
    np.testing.assert_array_equal(digits, expected)


def test_extract_leading_digits_zeros_excluded():
    """Zeros and tiny values are excluded."""
    weights = np.array([0.0, 1e-15, 1e-11, 1.5, 2.3])
    digits = extract_leading_digits(weights)
    assert len(digits) == 2  # only 1.5 and 2.3


def test_extract_leading_digits_empty():
    """Empty input returns empty array."""
    digits = extract_leading_digits(np.array([]))
    assert len(digits) == 0


def test_compute_distribution_sums_to_one():
    """Distribution sums to 1."""
    digits = np.array([1, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    dist = compute_digit_distribution(digits)
    total = sum(dist.values())
    assert abs(total - 1.0) < 1e-10


def test_compute_distribution_values():
    """Distribution correctly counts digit frequencies."""
    digits = np.array([1, 1, 1, 2, 2, 3])
    dist = compute_digit_distribution(digits)
    assert abs(dist[1] - 0.5) < 1e-10
    assert abs(dist[2] - 1.0 / 3) < 1e-10
    assert abs(dist[3] - 1.0 / 6) < 1e-10


def test_chi_squared_perfect_benford():
    """Perfect Benford distribution should give chi2 ~ 0."""
    expected = benford_expected()
    chi2, p_val, df = chi_squared_test(expected, 10000)
    assert chi2 < 0.001
    assert p_val > 0.99
    assert df == 8


def test_chi_squared_uniform():
    """Uniform distribution should give large chi2."""
    uniform = {d: 1.0 / 9 for d in range(1, 10)}
    chi2, p_val, df = chi_squared_test(uniform, 10000)
    assert chi2 > 100  # very different from Benford
    assert p_val < 0.001


def test_mad_perfect_benford():
    """Perfect Benford distribution should give MAD = 0."""
    expected = benford_expected()
    mad = mad_from_benford(expected)
    assert abs(mad) < 1e-10


def test_mad_uniform():
    """Uniform distribution should have large MAD."""
    uniform = {d: 1.0 / 9 for d in range(1, 10)}
    mad = mad_from_benford(uniform)
    assert mad > 0.015  # nonconformity


def test_classify_mad():
    """MAD classification thresholds."""
    assert classify_mad(0.003) == "close"
    assert classify_mad(0.008) == "acceptable"
    assert classify_mad(0.013) == "marginal"
    assert classify_mad(0.020) == "nonconformity"


def test_kl_divergence_self():
    """KL divergence of Benford with itself is 0."""
    expected = benford_expected()
    kl = kl_divergence(expected)
    assert abs(kl) < 1e-10


def test_kl_divergence_positive():
    """KL divergence is non-negative."""
    uniform = {d: 1.0 / 9 for d in range(1, 10)}
    kl = kl_divergence(uniform)
    assert kl > 0


def test_generate_control_weights():
    """Control weights generation produces all expected controls."""
    controls = generate_control_weights(n=1000, seed=42)
    assert "uniform" in controls
    assert "normal" in controls
    assert "kaiming_uniform" in controls
    for name, data in controls.items():
        assert "mad" in data
        assert "chi2" in data
        assert "p_value" in data
        assert data["n_weights"] > 0
