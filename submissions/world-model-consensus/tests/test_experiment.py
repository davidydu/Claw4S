"""Tests for src/experiment.py — simulation runner and config."""

import numpy as np
import pytest

from src.experiment import (
    SimulationConfig, run_simulation, get_composition,
    build_experiment_matrix, COMPOSITIONS,
)


class TestCompositions:
    def test_all_compositions_exist(self):
        expected = {"all_adaptive", "all_stubborn", "mixed", "leader_followers"}
        assert set(COMPOSITIONS.keys()) == expected

    def test_get_composition_length(self):
        for name in COMPOSITIONS:
            for n in [3, 4, 6]:
                comp = get_composition(name, n)
                assert len(comp) == n

    def test_leader_followers_has_one_leader(self):
        comp = get_composition("leader_followers", 4)
        assert comp.count("leader") == 1
        assert comp.count("follower") == 3

    def test_mixed_composition(self):
        comp = get_composition("mixed", 4)
        assert comp.count("adaptive") == 2
        assert comp.count("stubborn") == 2


class TestSimulation:
    def test_basic_run(self):
        cfg = SimulationConfig(n_agents=3, n_rounds=100, disagreement=0.0, seed=42)
        result = run_simulation(cfg)
        assert result.action_history.shape == (100, 3)
        assert result.payoff_history.shape == (100, 3)
        assert len(result.coordinated) == 100

    def test_zero_disagreement_adaptive_coordinates(self):
        cfg = SimulationConfig(
            n_agents=4, n_rounds=500, disagreement=0.0,
            composition="all_adaptive", seed=42,
        )
        result = run_simulation(cfg)
        # With zero disagreement and 5% epsilon, coordination ~ (0.95)^4 ~ 0.81
        tail = result.coordinated[int(500 * 0.8):]
        assert tail.mean() > 0.7, f"Expected high coordination at d=0, got {tail.mean()}"

    def test_stubborn_high_disagreement_fails(self):
        cfg = SimulationConfig(
            n_agents=4, n_rounds=500, disagreement=1.0,
            composition="all_stubborn", seed=42,
        )
        result = run_simulation(cfg)
        # Stubborn agents with different priors can't coordinate
        tail = result.coordinated[int(500 * 0.8):]
        assert tail.mean() < 0.5, f"Stubborn agents at d=1.0 should fail, got {tail.mean()}"

    def test_deterministic(self):
        cfg = SimulationConfig(n_agents=3, n_rounds=100, seed=42)
        r1 = run_simulation(cfg)
        r2 = run_simulation(cfg)
        np.testing.assert_array_equal(r1.action_history, r2.action_history)

    def test_payoffs_consistent_with_coordinated(self):
        cfg = SimulationConfig(n_agents=4, n_rounds=200, seed=42)
        result = run_simulation(cfg)
        for t in range(200):
            if result.coordinated[t]:
                assert np.all(result.payoff_history[t] == 1.0)
            else:
                assert np.all(result.payoff_history[t] == 0.0)


class TestExperimentMatrix:
    def test_matrix_size(self):
        configs = build_experiment_matrix()
        # 4 compositions x 11 disagreements x 3 sizes x 3 seeds = 396
        assert len(configs) == 396

    def test_all_configs_valid(self):
        configs = build_experiment_matrix()
        for cfg in configs:
            assert cfg.n_agents in [3, 4, 6]
            assert 0.0 <= cfg.disagreement <= 1.0
            assert cfg.composition in COMPOSITIONS
            assert cfg.n_rounds == 10_000
