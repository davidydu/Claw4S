"""Tests for ground-truth distributions and quality metrics."""

import numpy as np
import pytest

from src.distributions import (
    DISTRIBUTIONS,
    fit_kde,
    ground_truth_pdf,
    kl_divergence_numerical,
    sample_from_kde,
    sample_ground_truth,
    wasserstein,
)


class TestSampling:
    """Sampling from ground-truth distributions."""

    @pytest.mark.parametrize("dist_name", list(DISTRIBUTIONS.keys()))
    def test_sample_shape(self, dist_name: str) -> None:
        rng = np.random.default_rng(42)
        samples = sample_ground_truth(dist_name, 1000, rng)
        assert samples.shape == (1000,)

    @pytest.mark.parametrize("dist_name", list(DISTRIBUTIONS.keys()))
    def test_sample_reproducibility(self, dist_name: str) -> None:
        s1 = sample_ground_truth(dist_name, 500, np.random.default_rng(42))
        s2 = sample_ground_truth(dist_name, 500, np.random.default_rng(42))
        np.testing.assert_array_equal(s1, s2)

    def test_pdf_integrates_to_one(self) -> None:
        from scipy.integrate import quad
        for dist_name in DISTRIBUTIONS:
            val, _ = quad(lambda x: ground_truth_pdf(dist_name, np.array([x]))[0], -20, 20)
            assert abs(val - 1.0) < 1e-4, f"{dist_name} PDF integrates to {val}"


class TestKDE:
    """KDE fitting and sampling."""

    def test_fit_and_sample(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_ground_truth("bimodal", 2000, rng)
        kde = fit_kde(samples)
        resampled = sample_from_kde(kde, 500, rng)
        assert resampled.shape == (500,)

    def test_kde_close_to_truth(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_ground_truth("bimodal", 5000, rng)
        kde = fit_kde(samples)
        kl = kl_divergence_numerical("bimodal", kde)
        assert kl < 0.3, f"KDE from 5000 samples has KL={kl:.4f}, expected < 0.3"


class TestMetrics:
    """Quality metrics: KL divergence and Wasserstein distance."""

    def test_kl_nonnegative(self) -> None:
        rng = np.random.default_rng(42)
        samples = sample_ground_truth("skewed", 2000, rng)
        kde = fit_kde(samples)
        kl = kl_divergence_numerical("skewed", kde)
        assert kl >= 0.0

    def test_kl_symmetry(self) -> None:
        """Identical seeds must produce identical KL values."""
        for seed in [42, 123, 789]:
            rng1 = np.random.default_rng(seed)
            rng2 = np.random.default_rng(seed)
            s1 = sample_ground_truth("bimodal", 2000, rng1)
            s2 = sample_ground_truth("bimodal", 2000, rng2)
            kde1 = fit_kde(s1)
            kde2 = fit_kde(s2)
            kl1 = kl_divergence_numerical("bimodal", kde1)
            kl2 = kl_divergence_numerical("bimodal", kde2)
            assert kl1 == kl2, f"Seed {seed}: {kl1} != {kl2}"

    def test_wasserstein_same_dist(self) -> None:
        rng = np.random.default_rng(42)
        s1 = sample_ground_truth("bimodal", 5000, rng)
        s2 = sample_ground_truth("bimodal", 2000, rng)
        wd = wasserstein(s1, s2)
        assert wd < 0.5, f"Same-dist WD={wd:.4f} too large"

    def test_wasserstein_different_dists(self) -> None:
        rng = np.random.default_rng(42)
        s1 = sample_ground_truth("bimodal", 5000, rng)
        s2 = sample_ground_truth("skewed", 5000, np.random.default_rng(42))
        wd = wasserstein(s1, s2)
        assert wd > 0.5, f"Cross-dist WD={wd:.4f} too small"
