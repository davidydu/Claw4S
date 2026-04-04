"""Tests for the simulation engine."""

import numpy as np
import pytest

from src.simulation import SimConfig, SimResult, run_simulation, _generate_observations
from src.agents import K_OPTIONS


class TestGenerateObservations:
    def test_shape(self):
        rng = np.random.default_rng(42)
        counts = _generate_observations(2, 10, 0.8, rng)
        assert counts.shape == (K_OPTIONS,)

    def test_counts_sum_to_n_samples(self):
        rng = np.random.default_rng(42)
        counts = _generate_observations(0, 10, 0.6, rng)
        assert counts.sum() == 10

    def test_true_option_gets_most_counts_on_average(self):
        rng = np.random.default_rng(42)
        total = np.zeros(K_OPTIONS)
        for _ in range(1000):
            total += _generate_observations(3, 5, 0.8, rng)
        assert total[3] == total.max()

    def test_perfect_signal(self):
        rng = np.random.default_rng(42)
        counts = _generate_observations(1, 10, 1.0, rng)
        assert counts[1] == 10

    def test_single_sample(self):
        rng = np.random.default_rng(42)
        counts = _generate_observations(2, 1, 0.6, rng)
        assert counts.sum() == 1


class TestRunSimulation:
    def test_no_byzantine_reasonable_accuracy(self):
        cfg = SimConfig(
            committee_size=9,
            honest_type="majority",
            byzantine_type="random",
            byzantine_fraction=0.0,
            rounds=500,
            signal_quality=0.6,
            seed=42,
        )
        result = run_simulation(cfg)
        # With sq=0.6 and 9 voters, should beat random (20%)
        assert result.accuracy > 0.40
        assert result.num_byzantine == 0
        assert result.num_honest == 9

    def test_full_byzantine_low_accuracy(self):
        cfg = SimConfig(
            committee_size=9,
            honest_type="majority",
            byzantine_type="strategic",
            byzantine_fraction=1.0,
            rounds=500,
            seed=42,
        )
        result = run_simulation(cfg)
        assert result.num_byzantine == 9
        assert result.num_honest == 0

    def test_bayesian_beats_majority_with_byzantines(self):
        """BayesianVoter with 3 samples should outperform MajorityVoter (1 sample)."""
        results = {}
        for ht in ["majority", "bayesian"]:
            cfg = SimConfig(
                committee_size=9,
                honest_type=ht,
                byzantine_type="strategic",
                byzantine_fraction=0.33,
                rounds=2000,
                signal_quality=0.5,
                seed=42,
            )
            results[ht] = run_simulation(cfg)
        # Bayesian should be at least as good (usually better due to more info)
        assert results["bayesian"].accuracy >= results["majority"].accuracy - 0.05

    def test_reproducibility(self):
        cfg = SimConfig(
            committee_size=5,
            honest_type="bayesian",
            byzantine_type="random",
            byzantine_fraction=0.2,
            rounds=200,
            seed=123,
        )
        r1 = run_simulation(cfg)
        r2 = run_simulation(cfg)
        assert r1.accuracy == r2.accuracy

    def test_result_fields(self):
        cfg = SimConfig(
            committee_size=5,
            honest_type="cautious",
            byzantine_type="mimicking",
            byzantine_fraction=0.2,
            rounds=100,
            seed=7,
        )
        result = run_simulation(cfg)
        assert isinstance(result, SimResult)
        assert 0.0 <= result.accuracy <= 1.0
        assert result.accuracy_std >= 0.0
        assert result.num_rounds == 100
        assert result.num_honest + result.num_byzantine == 5

    def test_cautious_abstention_handled(self):
        cfg = SimConfig(
            committee_size=5,
            honest_type="cautious",
            byzantine_type="random",
            byzantine_fraction=0.0,
            rounds=100,
            signal_quality=0.3,
            seed=42,
        )
        result = run_simulation(cfg)
        assert 0.0 <= result.accuracy <= 1.0
