"""Tests for the simulation runner."""

from src.simulation import run_single_sim


def test_baseline_simulation():
    result = run_single_sim(
        n_honest=10, n_sybil=0, algorithm_name="simple_average",
        strategy_name="none", n_rounds=100, seed=42,
    )
    assert "config" in result
    assert "metrics" in result
    assert result["config"]["n_sybil"] == 0
    assert result["metrics"]["sybil_detection_rate"] == 1.0


def test_sybil_simulation():
    result = run_single_sim(
        n_honest=10, n_sybil=5, algorithm_name="eigentrust",
        strategy_name="ballot_stuffing", n_rounds=200, seed=42,
    )
    assert result["config"]["n_sybil"] == 5
    assert len(result["sybil_reputations"]) == 5
    assert len(result["honest_reputations"]) == 10


def test_deterministic_across_runs():
    r1 = run_single_sim(
        n_honest=10, n_sybil=3, algorithm_name="pagerank_trust",
        strategy_name="bad_mouthing", n_rounds=100, seed=99,
    )
    r2 = run_single_sim(
        n_honest=10, n_sybil=3, algorithm_name="pagerank_trust",
        strategy_name="bad_mouthing", n_rounds=100, seed=99,
    )
    assert r1["metrics"] == r2["metrics"]


def test_all_algorithms_run():
    for algo in ["simple_average", "weighted_history", "pagerank_trust", "eigentrust"]:
        result = run_single_sim(
            n_honest=10, n_sybil=2, algorithm_name=algo,
            strategy_name="whitewashing", n_rounds=100, seed=42,
        )
        assert "metrics" in result


def test_all_strategies_run():
    for strat in ["ballot_stuffing", "bad_mouthing", "whitewashing"]:
        result = run_single_sim(
            n_honest=10, n_sybil=3, algorithm_name="simple_average",
            strategy_name=strat, n_rounds=100, seed=42,
        )
        assert "metrics" in result
