"""Tests for norm emergence metrics."""

import numpy as np

from src.metrics import (
    CONVERGENCE_THRESHOLD,
    WINDOW_SIZE,
    norm_convergence_time,
    norm_diversity,
    norm_efficiency,
    norm_fragility,
)


def test_convergence_already_converged():
    """All-same-action history converges at WINDOW_SIZE."""
    history = np.zeros(2000, dtype=np.int32)
    result = norm_convergence_time(history, 2000)
    assert result == WINDOW_SIZE


def test_convergence_never_converges():
    """Uniformly random actions should not converge."""
    rng = np.random.default_rng(42)
    history = rng.integers(0, 3, size=5000).astype(np.int32)
    result = norm_convergence_time(history, 5000)
    assert result == 5000


def test_convergence_midway():
    """History that starts mixed then converges mid-simulation."""
    rng = np.random.default_rng(42)
    mixed = rng.integers(0, 3, size=1000).astype(np.int32)
    converged = np.ones(2000, dtype=np.int32)  # all action 1
    history = np.concatenate([mixed, converged])
    result = norm_convergence_time(history, len(history))
    # Should converge sometime after the mixed phase
    assert WINDOW_SIZE < result <= 1000 + WINDOW_SIZE


def test_efficiency_perfect():
    """Perfect coordination yields efficiency 1.0."""
    payoffs = np.full(1000, 5.0)
    eff = norm_efficiency(payoffs, 5.0)
    assert eff == 1.0


def test_efficiency_zero_payoff():
    """Zero payoffs yield efficiency 0.0."""
    payoffs = np.zeros(1000)
    eff = norm_efficiency(payoffs, 5.0)
    assert eff == 0.0


def test_efficiency_partial():
    """Half-optimal payoffs yield ~0.5 efficiency."""
    payoffs = np.full(1000, 2.5)
    eff = norm_efficiency(payoffs, 5.0)
    assert abs(eff - 0.5) < 0.01


def test_diversity_single_norm():
    """All-same-action history has diversity 1."""
    history = np.zeros(1000, dtype=np.int32)
    assert norm_diversity(history) == 1


def test_diversity_two_norms():
    """Two interleaved actions give diversity 2."""
    # Interleave so the tail contains both actions
    history = np.array([0, 1] * 500, dtype=np.int32)
    assert norm_diversity(history) == 2


def test_diversity_three_norms():
    """Three interleaved actions give diversity 3."""
    history = np.array([0, 1, 2] * 334, dtype=np.int32)
    assert norm_diversity(history) == 3


def test_fragility_stable_norm():
    """When all innovator fractions keep the same norm, fragility = 1.0."""
    baseline = np.zeros(1000, dtype=np.int32)
    fractions = [0.1, 0.2, 0.3]
    histories = [np.zeros(1000, dtype=np.int32) for _ in fractions]
    frag = norm_fragility(baseline, fractions, histories)
    assert frag == 1.0


def test_fragility_breaks_immediately():
    """When lowest innovator fraction shifts norm, fragility = 0.1."""
    baseline = np.zeros(1000, dtype=np.int32)
    fractions = [0.1, 0.2, 0.3]
    histories = [np.ones(1000, dtype=np.int32) for _ in fractions]
    frag = norm_fragility(baseline, fractions, histories)
    assert frag == 0.1
