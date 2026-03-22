"""Tests for cascade metrics computation."""

from src.metrics import (
    cascade_formation_rate,
    cascade_accuracy,
    cascade_fragility,
    mean_cascade_length,
    majority_accuracy,
    compute_group_metrics,
    compute_standard_error,
    proportion_ci_95,
)


def _make_result(cascade_formed, cascade_correct=None, cascade_length=0,
                 cascade_start=None, n_agents=10, majority_correct=True):
    """Helper to create a minimal simulation result dict."""
    return {
        "cascade_formed": cascade_formed,
        "cascade_correct": cascade_correct,
        "cascade_length": cascade_length,
        "cascade_start": cascade_start,
        "n_agents": n_agents,
        "majority_correct": majority_correct,
    }


def test_cascade_formation_rate_all_formed():
    results = [_make_result(True) for _ in range(10)]
    assert cascade_formation_rate(results) == 1.0


def test_cascade_formation_rate_none_formed():
    results = [_make_result(False) for _ in range(10)]
    assert cascade_formation_rate(results) == 0.0


def test_cascade_formation_rate_half():
    results = [_make_result(True) for _ in range(5)] + [_make_result(False) for _ in range(5)]
    assert cascade_formation_rate(results) == 0.5


def test_cascade_formation_rate_empty():
    assert cascade_formation_rate([]) == 0.0


def test_cascade_accuracy_all_correct():
    results = [_make_result(True, cascade_correct=True) for _ in range(10)]
    assert cascade_accuracy(results) == 1.0


def test_cascade_accuracy_half():
    results = (
        [_make_result(True, cascade_correct=True) for _ in range(5)]
        + [_make_result(True, cascade_correct=False) for _ in range(5)]
    )
    assert cascade_accuracy(results) == 0.5


def test_cascade_accuracy_no_cascades():
    results = [_make_result(False) for _ in range(10)]
    assert cascade_accuracy(results) is None


def test_cascade_fragility_none_broken():
    # cascade starts at 3, length 7, n_agents=10 => remaining=7, not broken
    results = [_make_result(True, cascade_start=3, cascade_length=7, n_agents=10)]
    assert cascade_fragility(results) == 0.0


def test_cascade_fragility_all_broken():
    # cascade starts at 3, length 2, n_agents=10 => remaining=7, broken
    results = [_make_result(True, cascade_start=3, cascade_length=2, n_agents=10)]
    assert cascade_fragility(results) == 1.0


def test_cascade_fragility_no_cascades():
    results = [_make_result(False)]
    assert cascade_fragility(results) is None


def test_mean_cascade_length_basic():
    results = [
        _make_result(True, cascade_length=5),
        _make_result(True, cascade_length=10),
        _make_result(True, cascade_length=3),
    ]
    assert mean_cascade_length(results) == 6.0


def test_mean_cascade_length_no_cascades():
    results = [_make_result(False)]
    assert mean_cascade_length(results) is None


def test_majority_accuracy():
    results = [
        _make_result(False, majority_correct=True),
        _make_result(False, majority_correct=False),
        _make_result(False, majority_correct=True),
    ]
    assert abs(majority_accuracy(results) - 2 / 3) < 1e-9


def test_compute_group_metrics_keys():
    results = [_make_result(True, cascade_correct=True, cascade_length=5,
                            cascade_start=2, n_agents=10)]
    metrics = compute_group_metrics(results)
    expected_keys = {
        "n_simulations", "cascade_formation_rate", "cascade_accuracy",
        "cascade_fragility", "mean_cascade_length", "majority_accuracy",
    }
    assert set(metrics.keys()) == expected_keys


def test_compute_standard_error():
    values = [1.0, 1.0, 1.0]
    assert compute_standard_error(values) == 0.0

    values = [0.0, 1.0]
    se = compute_standard_error(values)
    assert se > 0


def test_proportion_ci_95_zero():
    lo, hi = proportion_ci_95(0, 0)
    assert lo == 0.0
    assert hi == 0.0


def test_proportion_ci_95_all():
    lo, hi = proportion_ci_95(100, 100)
    assert lo > 0.95
    assert hi > 0.99


def test_proportion_ci_95_half():
    lo, hi = proportion_ci_95(50, 100)
    assert 0.35 < lo < 0.50
    assert 0.50 < hi < 0.65
