"""Tests for the simulation engine."""

import numpy as np

from src.agents import AgentType
from src.game import make_symmetric_game
from src.simulation import compute_sim_metrics, run_simulation


def test_run_simulation_output_shapes():
    """run_simulation returns correct array shapes."""
    game = make_symmetric_game()
    comp = {AgentType.ADAPTIVE: 10}
    result = run_simulation(game, comp, total_rounds=500, seed=42)

    assert result["action_history"].shape == (500,)
    assert result["payoff_history"].shape == (500,)
    assert len(result["agents"]) == 10


def test_run_simulation_deterministic():
    """Same seed produces identical results."""
    game = make_symmetric_game()
    comp = {AgentType.ADAPTIVE: 10}
    r1 = run_simulation(game, comp, total_rounds=500, seed=42)
    r2 = run_simulation(game, comp, total_rounds=500, seed=42)

    np.testing.assert_array_equal(r1["action_history"], r2["action_history"])
    np.testing.assert_array_equal(r1["payoff_history"], r2["payoff_history"])


def test_run_simulation_different_seeds():
    """Different seeds produce different results."""
    game = make_symmetric_game()
    comp = {AgentType.ADAPTIVE: 10}
    r1 = run_simulation(game, comp, total_rounds=500, seed=42)
    r2 = run_simulation(game, comp, total_rounds=500, seed=99)

    # Extremely unlikely to be identical
    assert not np.array_equal(r1["action_history"], r2["action_history"])


def test_compute_sim_metrics_keys():
    """compute_sim_metrics returns all expected keys."""
    game = make_symmetric_game()
    comp = {AgentType.ADAPTIVE: 10}
    metrics = compute_sim_metrics(game, comp, total_rounds=1000, seed=42)

    expected_keys = {
        "game", "composition", "population_size", "total_rounds", "seed",
        "convergence_time", "efficiency", "diversity", "fragility",
    }
    assert expected_keys.issubset(set(metrics.keys()))


def test_compute_sim_metrics_ranges():
    """All metrics fall within valid ranges."""
    game = make_symmetric_game()
    comp = {AgentType.ADAPTIVE: 10}
    metrics = compute_sim_metrics(game, comp, total_rounds=1000, seed=42)

    assert 0.0 <= metrics["efficiency"] <= 1.0
    assert metrics["diversity"] in (0, 1, 2, 3)
    assert 0 <= metrics["convergence_time"] <= 1000
    assert 0.0 <= metrics["fragility"] <= 1.0


def test_actions_in_valid_range():
    """All recorded actions are in [0, 2]."""
    game = make_symmetric_game()
    comp = {AgentType.ADAPTIVE: 5, AgentType.INNOVATOR: 5}
    result = run_simulation(game, comp, total_rounds=500, seed=42)
    assert np.all(result["action_history"] >= 0)
    assert np.all(result["action_history"] <= 2)
