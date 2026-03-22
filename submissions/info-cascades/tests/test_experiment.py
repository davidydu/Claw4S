"""Tests for experiment runner."""

from src.experiment import build_task_list, group_results, compute_all_metrics
from src.simulation import run_single_simulation


def test_build_task_list_count():
    """Task list should have 4 types x 3 qualities x 3 lengths x 2 states x 3 seeds = 216."""
    tasks = build_task_list()
    assert len(tasks) == 216


def test_build_task_list_structure():
    """Each task is a 5-tuple."""
    tasks = build_task_list()
    for t in tasks:
        assert len(t) == 5
        agent_type, n_agents, signal_quality, true_state, seed = t
        assert isinstance(agent_type, str)
        assert isinstance(n_agents, int)
        assert isinstance(signal_quality, float)
        assert true_state in (0, 1)
        assert isinstance(seed, int)


def test_group_results():
    """group_results creates groups by (type, quality, n_agents)."""
    results = [
        run_single_simulation("bayesian", 10, 0.7, 0, 42),
        run_single_simulation("bayesian", 10, 0.7, 1, 42),
        run_single_simulation("bayesian", 20, 0.7, 0, 42),
    ]
    groups = group_results(results)
    assert len(groups) == 2  # (bayesian, 0.7, 10) and (bayesian, 0.7, 20)
    assert len(groups[("bayesian", 0.7, 10)]) == 2
    assert len(groups[("bayesian", 0.7, 20)]) == 1


def test_compute_all_metrics_structure():
    """compute_all_metrics returns correct keys for each group."""
    results = [
        run_single_simulation("bayesian", 10, 0.7, 0, 42),
        run_single_simulation("bayesian", 10, 0.7, 1, 123),
    ]
    metrics = compute_all_metrics(results)
    assert len(metrics) == 1
    m = metrics[0]
    assert m["agent_type"] == "bayesian"
    assert m["signal_quality"] == 0.7
    assert m["n_agents"] == 10
    assert m["n_simulations"] == 2
    assert "cascade_formation_rate" in m
    assert "cascade_accuracy" in m
    assert "cascade_fragility" in m
    assert "mean_cascade_length" in m
    assert "majority_accuracy" in m
