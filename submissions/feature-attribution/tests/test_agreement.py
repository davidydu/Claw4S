"""Tests for agreement measurement."""

import numpy as np
from src.agreement import pairwise_spearman, aggregate_agreement


def test_pairwise_spearman_perfect_agreement():
    """Identical attributions should yield rho = 1.0."""
    attrs = {
        "method_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "method_b": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    result = pairwise_spearman(attrs, [("method_a", "method_b")])
    assert abs(result["method_a_vs_method_b"] - 1.0) < 1e-6


def test_pairwise_spearman_perfect_disagreement():
    """Reversed attributions should yield rho = -1.0."""
    attrs = {
        "method_a": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "method_b": np.array([5.0, 4.0, 3.0, 2.0, 1.0]),
    }
    result = pairwise_spearman(attrs, [("method_a", "method_b")])
    assert abs(result["method_a_vs_method_b"] + 1.0) < 1e-6


def test_pairwise_spearman_handles_constant():
    """Constant attributions (all zero) should return 0.0, not NaN."""
    attrs = {
        "method_a": np.zeros(5),
        "method_b": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    result = pairwise_spearman(attrs, [("method_a", "method_b")])
    assert result["method_a_vs_method_b"] == 0.0


def test_aggregate_agreement():
    """Aggregation computes correct mean and std."""
    samples = [
        {"pair_a": 0.8, "pair_b": 0.5},
        {"pair_a": 0.6, "pair_b": 0.7},
        {"pair_a": 0.7, "pair_b": 0.6},
    ]
    agg = aggregate_agreement(samples)
    assert "pair_a" in agg
    assert "pair_b" in agg

    np.testing.assert_almost_equal(agg["pair_a"]["mean"], 0.7, decimal=5)
    np.testing.assert_almost_equal(agg["pair_b"]["mean"], 0.6, decimal=5)

    # std should be > 0
    assert agg["pair_a"]["std"] > 0
    assert agg["pair_b"]["std"] > 0


def test_aggregate_agreement_empty_raises():
    """Aggregation should raise ValueError for empty input."""
    try:
        aggregate_agreement([])
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
