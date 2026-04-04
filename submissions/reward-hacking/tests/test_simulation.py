"""Tests for the core simulation engine."""

import numpy as np
import pytest

from src.agents import create_agent_population
from src.network import build_adjacency, neighbor_list
from src.simulation import run_simulation, sigmoid


class TestSigmoid:
    def test_zero(self):
        assert abs(sigmoid(0.0) - 0.5) < 1e-10

    def test_large_positive(self):
        assert abs(sigmoid(100.0) - 1.0) < 1e-10

    def test_large_negative(self):
        assert abs(sigmoid(-100.0) - 0.0) < 1e-10

    def test_symmetry(self):
        assert abs(sigmoid(2.0) + sigmoid(-2.0) - 1.0) < 1e-10


class TestRunSimulation:
    def _make_setup(self, n=10, monitor_frac=0.0, topology="grid", seed=42):
        rng = np.random.default_rng(seed)
        agents = create_agent_population(n, monitor_frac, rng)
        adj = build_adjacency(n, topology, rng)
        nbrs = neighbor_list(adj)
        return agents, nbrs, rng

    def test_returns_correct_keys(self):
        agents, nbrs, rng = self._make_setup()
        result = run_simulation(agents, nbrs, 100, [0], 10, "obvious", rng)
        expected_keys = {
            "adoption_curve", "proxy_reward_curve", "true_reward_curve",
            "divergence_curve", "containment_events", "final_adoption",
            "time_to_50pct", "time_to_90pct",
        }
        assert set(result.keys()) == expected_keys

    def test_adoption_curve_length(self):
        agents, nbrs, rng = self._make_setup()
        result = run_simulation(agents, nbrs, 200, [0], 10, "obvious", rng)
        assert len(result["adoption_curve"]) == 200

    def test_no_adoption_before_discovery(self):
        agents, nbrs, rng = self._make_setup()
        result = run_simulation(agents, nbrs, 100, [0], 50, "obvious", rng)
        # Before round 50, no hacking should occur
        for t in range(50):
            assert result["adoption_curve"][t] == 0.0

    def test_hack_spreads_without_monitors(self):
        agents, nbrs, rng = self._make_setup(monitor_frac=0.0, seed=42)
        result = run_simulation(agents, nbrs, 2000, [0, 1], 10, "invisible", rng)
        # With invisible hack and no monitors, hack should spread
        assert result["final_adoption"] > 0.0

    def test_monitors_reduce_adoption(self):
        # Without monitors
        a1, n1, r1 = self._make_setup(monitor_frac=0.0, seed=42)
        res1 = run_simulation(a1, n1, 1000, [0, 1], 10, "obvious", r1)

        # With 50% monitors
        a2, n2, r2 = self._make_setup(monitor_frac=0.5, seed=42)
        res2 = run_simulation(a2, n2, 1000, [0, 1], 10, "obvious", r2)

        # Monitors should reduce or maintain lower adoption
        assert res2["final_adoption"] <= res1["final_adoption"] + 0.3

    def test_deterministic(self):
        a1, n1, r1 = self._make_setup(seed=42)
        res1 = run_simulation(a1, n1, 100, [0], 10, "obvious", r1)

        a2, n2, r2 = self._make_setup(seed=42)
        res2 = run_simulation(a2, n2, 100, [0], 10, "obvious", r2)

        assert res1["adoption_curve"] == res2["adoption_curve"]

    def test_adoption_bounded(self):
        agents, nbrs, rng = self._make_setup()
        result = run_simulation(agents, nbrs, 500, [0], 10, "obvious", rng)
        for frac in result["adoption_curve"]:
            assert 0.0 <= frac <= 1.0


class TestSimulationEdgeCases:
    def test_single_agent(self):
        rng = np.random.default_rng(42)
        agents = create_agent_population(1, 0.0, rng)
        adj = build_adjacency(1, "grid", rng)
        nbrs = neighbor_list(adj)
        result = run_simulation(agents, nbrs, 100, [0], 10, "obvious", rng)
        # Single agent discovers hack, stays hacking
        assert result["adoption_curve"][10] == 1.0

    def test_all_monitors(self):
        rng = np.random.default_rng(42)
        agents = create_agent_population(5, 1.0, rng)
        adj = build_adjacency(5, "grid", rng)
        nbrs = neighbor_list(adj)
        # No non-monitor agents to be initial hackers
        result = run_simulation(agents, nbrs, 100, [], 10, "obvious", rng)
        assert result["final_adoption"] == 0.0
