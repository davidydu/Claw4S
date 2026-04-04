"""Tests for the experiment runner."""

import pytest

from src.experiment import _build_configs, _run_one, _aggregate_results
from src.simulation import SimConfig


class TestBuildConfigs:
    def test_total_count(self):
        configs = _build_configs()
        # 3 honest x 3 byzantine x 5 fractions x 3 sizes x 3 seeds = 405
        assert len(configs) == 405

    def test_all_configs_are_simconfig(self):
        configs = _build_configs()
        assert all(isinstance(c, SimConfig) for c in configs)

    def test_fractions_present(self):
        configs = _build_configs()
        fracs = sorted(set(round(c.byzantine_fraction, 4) for c in configs))
        assert len(fracs) == 5
        assert fracs[0] == 0.0
        assert fracs[-1] == 0.5


class TestRunOne:
    def test_returns_dict(self):
        cfg = SimConfig(
            committee_size=5,
            honest_type="majority",
            byzantine_type="random",
            byzantine_fraction=0.2,
            rounds=50,
            seed=42,
        )
        result = _run_one(cfg)
        assert isinstance(result, dict)
        assert "accuracy" in result
        assert "committee_size" in result

    def test_accuracy_in_range(self):
        cfg = SimConfig(
            committee_size=9,
            honest_type="bayesian",
            byzantine_type="strategic",
            byzantine_fraction=0.33,
            rounds=50,
            seed=42,
        )
        result = _run_one(cfg)
        assert 0.0 <= result["accuracy"] <= 1.0


class TestAggregateResults:
    def test_aggregation_structure(self):
        raw = [
            {"honest_type": "majority", "byzantine_type": "random",
             "byzantine_fraction": 0.0, "committee_size": 5,
             "seed": 42, "rounds": 50, "accuracy": 0.80,
             "accuracy_std": 0.01, "num_honest": 5, "num_byzantine": 0},
            {"honest_type": "majority", "byzantine_type": "random",
             "byzantine_fraction": 0.0, "committee_size": 5,
             "seed": 123, "rounds": 50, "accuracy": 0.82,
             "accuracy_std": 0.01, "num_honest": 5, "num_byzantine": 0},
        ]
        agg = _aggregate_results(raw)
        assert "summaries" in agg
        assert "derived_metrics" in agg
        assert len(agg["summaries"]) == 1  # same group, averaged
        assert abs(agg["summaries"][0]["mean_accuracy"] - 0.81) < 1e-6
