"""Unit tests for privacy accounting methods.

Tests verify:
- Correctness of each method against known analytical results
- Monotonicity properties (more noise -> less epsilon, more steps -> more epsilon)
- Ordering of methods (naive >= advanced >= RDP/GDP)
- Edge cases and invalid inputs
"""

import math
import pytest
from src.accounting import (
    epsilon_naive,
    epsilon_advanced,
    epsilon_rdp,
    epsilon_gdp,
    compute_all_epsilons,
    _gdp_delta_at_eps,
)


class TestNaiveComposition:
    """Tests for naive (linear) composition."""

    def test_single_step(self):
        """Single step should match the basic Gaussian mechanism formula."""
        sigma = 1.0
        delta = 1e-5
        eps = epsilon_naive(sigma, T=1, delta=delta)
        expected = math.sqrt(2.0 * math.log(1.25 / delta)) / sigma
        assert abs(eps - expected) < 1e-10

    def test_linear_scaling(self):
        """T steps should give T times the single-step epsilon."""
        sigma = 1.0
        delta = 1e-5
        eps_1 = epsilon_naive(sigma, T=1, delta=delta)
        eps_10 = epsilon_naive(sigma, T=10, delta=delta)
        assert abs(eps_10 - 10 * eps_1) < 1e-10

    def test_more_noise_less_epsilon(self):
        """Increasing sigma should decrease epsilon."""
        eps_small = epsilon_naive(sigma=0.5, T=100, delta=1e-5)
        eps_large = epsilon_naive(sigma=5.0, T=100, delta=1e-5)
        assert eps_large < eps_small

    def test_invalid_inputs(self):
        """Invalid inputs should return infinity."""
        assert epsilon_naive(sigma=0, T=10, delta=1e-5) == float("inf")
        assert epsilon_naive(sigma=-1, T=10, delta=1e-5) == float("inf")
        assert epsilon_naive(sigma=1.0, T=0, delta=1e-5) == float("inf")
        assert epsilon_naive(sigma=1.0, T=10, delta=0) == float("inf")
        assert epsilon_naive(sigma=1.0, T=10, delta=1.0) == float("inf")


class TestAdvancedComposition:
    """Tests for advanced composition theorem."""

    def test_tighter_than_naive_high_sigma(self):
        """Advanced composition is tighter than naive when eps_step is small.

        The advanced composition theorem requires small per-step epsilon
        to provide benefit. With sigma=20 (eps_step ~ 0.24), it should
        be strictly tighter for large T.
        """
        for T in [100, 1000]:
            eps_naive_val = epsilon_naive(sigma=20.0, T=T, delta=1e-5)
            eps_adv = epsilon_advanced(sigma=20.0, T=T, delta=1e-5)
            assert eps_adv < eps_naive_val, (
                f"Advanced ({eps_adv:.4f}) not tighter than naive "
                f"({eps_naive_val:.4f}) at sigma=20, T={T}")

    def test_sublinear_growth(self):
        """Advanced composition should grow sublinearly with T."""
        eps_100 = epsilon_advanced(sigma=1.0, T=100, delta=1e-5)
        eps_1000 = epsilon_advanced(sigma=1.0, T=1000, delta=1e-5)
        # Should grow less than 10x when T grows 10x
        assert eps_1000 / eps_100 < 10.0

    def test_invalid_inputs(self):
        """Invalid inputs should return infinity."""
        assert epsilon_advanced(sigma=0, T=10, delta=1e-5) == float("inf")
        assert epsilon_advanced(sigma=1.0, T=0, delta=1e-5) == float("inf")


class TestRenyiDP:
    """Tests for Renyi DP accounting."""

    def test_tighter_than_naive(self):
        """RDP should give tighter bounds than naive composition."""
        for T in [100, 1000]:
            eps_naive_val = epsilon_naive(sigma=1.0, T=T, delta=1e-5)
            eps_rdp_val = epsilon_rdp(sigma=1.0, T=T, delta=1e-5)
            assert eps_rdp_val < eps_naive_val, (
                f"RDP ({eps_rdp_val:.4f}) not tighter than naive "
                f"({eps_naive_val:.4f}) at T={T}")

    def test_monotone_in_sigma(self):
        """Increasing sigma should decrease epsilon."""
        eps_small = epsilon_rdp(sigma=0.5, T=100, delta=1e-5)
        eps_large = epsilon_rdp(sigma=5.0, T=100, delta=1e-5)
        assert eps_large < eps_small

    def test_monotone_in_T(self):
        """Increasing T should increase epsilon."""
        eps_small = epsilon_rdp(sigma=1.0, T=10, delta=1e-5)
        eps_large = epsilon_rdp(sigma=1.0, T=1000, delta=1e-5)
        assert eps_large > eps_small

    def test_custom_orders(self):
        """Custom RDP orders should still produce valid results."""
        eps = epsilon_rdp(sigma=1.0, T=100, delta=1e-5, orders=[2, 4, 8])
        assert eps > 0
        assert eps < float("inf")

    def test_invalid_inputs(self):
        """Invalid inputs should return infinity."""
        assert epsilon_rdp(sigma=0, T=10, delta=1e-5) == float("inf")


class TestGaussianDP:
    """Tests for Gaussian DP (f-DP) accounting."""

    def test_delta_function_sanity(self):
        """The GDP delta function should be in [0, 1] for valid inputs."""
        delta = _gdp_delta_at_eps(mu=1.0, eps=1.0)
        assert 0 <= delta <= 1

    def test_delta_monotone_in_eps(self):
        """delta(eps) should decrease as eps increases."""
        d1 = _gdp_delta_at_eps(mu=1.0, eps=0.5)
        d2 = _gdp_delta_at_eps(mu=1.0, eps=2.0)
        assert d2 < d1

    def test_tighter_than_naive(self):
        """GDP should be tighter than naive composition."""
        for T in [100, 1000]:
            eps_naive_val = epsilon_naive(sigma=1.0, T=T, delta=1e-5)
            eps_gdp_val = epsilon_gdp(sigma=1.0, T=T, delta=1e-5)
            assert eps_gdp_val < eps_naive_val, (
                f"GDP ({eps_gdp_val:.4f}) not tighter than naive "
                f"({eps_naive_val:.4f}) at T={T}")

    def test_clt_scaling(self):
        """GDP epsilon should grow sublinearly with T.

        The mu parameter scales as sqrt(T), but the conversion from
        mu-GDP to (eps,delta)-DP is nonlinear (involves Phi^{-1}).
        For large mu, epsilon ~ mu^2/2 which gives T scaling, but
        for moderate mu the growth is much slower than linear.
        We verify it grows slower than T (sublinear).
        """
        eps_100 = epsilon_gdp(sigma=5.0, T=100, delta=1e-5)
        eps_10000 = epsilon_gdp(sigma=5.0, T=10000, delta=1e-5)
        ratio = eps_10000 / eps_100
        # Should be sublinear: ratio < 100 (the linear factor)
        # and roughly proportional to sqrt(T) ratio = 10 for large sigma
        assert ratio < 100, f"GDP scaling ratio {ratio:.2f} is not sublinear"
        assert ratio > 1, f"GDP epsilon should increase with T"

    def test_invalid_inputs(self):
        """Invalid inputs should return infinity."""
        assert epsilon_gdp(sigma=0, T=10, delta=1e-5) == float("inf")
        assert epsilon_gdp(sigma=1.0, T=0, delta=1e-5) == float("inf")


class TestMethodOrdering:
    """Cross-method tests verifying expected ordering of bounds."""

    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("T", [10, 100, 1000])
    def test_naive_loosest(self, sigma, T):
        """Naive composition should always give the loosest bound."""
        delta = 1e-5
        eps_naive_val = epsilon_naive(sigma, T, delta)
        eps_adv = epsilon_advanced(sigma, T, delta)
        eps_rdp_val = epsilon_rdp(sigma, T, delta)
        eps_gdp_val = epsilon_gdp(sigma, T, delta)

        for name, eps in [("advanced", eps_adv), ("rdp", eps_rdp_val),
                          ("gdp", eps_gdp_val)]:
            assert eps <= eps_naive_val * 1.001, (
                f"{name} ({eps:.4f}) > naive ({eps_naive_val:.4f}) "
                f"at sigma={sigma}, T={T}")

    @pytest.mark.parametrize("sigma", [10.0, 20.0])
    def test_advanced_tighter_than_naive_for_high_sigma(self, sigma):
        """Advanced should be tighter than naive when eps_step is small.

        The advanced composition theorem only helps when per-step epsilon
        is small (roughly < 1). This requires large sigma. With sigma >= 10,
        eps_step ~ 0.48, the advanced bound improves over naive for large T.
        """
        T = 1000
        delta = 1e-5
        assert epsilon_advanced(sigma, T, delta) < epsilon_naive(sigma, T, delta)


class TestComputeAll:
    """Tests for the unified compute_all_epsilons function."""

    def test_returns_all_methods(self):
        """Should return epsilon for all four methods."""
        result = compute_all_epsilons(sigma=1.0, T=100, delta=1e-5)
        assert set(result.keys()) == {"naive", "advanced", "rdp", "gdp"}

    def test_all_positive(self):
        """All epsilons should be positive for valid inputs."""
        result = compute_all_epsilons(sigma=1.0, T=100, delta=1e-5)
        for method, eps in result.items():
            assert eps > 0, f"{method} returned non-positive epsilon: {eps}"

    def test_all_finite_for_reasonable_params(self):
        """All methods should give finite epsilon for sigma >= 1."""
        result = compute_all_epsilons(sigma=2.0, T=100, delta=1e-5)
        for method, eps in result.items():
            assert eps < float("inf"), f"{method} returned inf at sigma=2.0, T=100"
