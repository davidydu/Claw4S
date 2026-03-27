"""Tests for DataEnvironment."""

import numpy as np
import pytest

from src.environment import DataEnvironment


class TestDataEnvironment:
    """Core environment tests."""

    def test_true_dist_sums_to_one(self):
        env = DataEnvironment(n_states=5, rng=np.random.default_rng(0))
        assert abs(env.true_dist.sum() - 1.0) < 1e-10

    def test_custom_dist(self):
        d = np.array([0.1, 0.2, 0.3, 0.4])
        env = DataEnvironment(n_states=4, true_dist=d)
        np.testing.assert_allclose(env.true_dist, d)

    def test_sample_true_shape(self):
        env = DataEnvironment(n_states=5, rng=np.random.default_rng(1))
        s = env.sample_true(100)
        assert s.shape == (100,)
        assert all(0 <= x < 5 for x in s)

    def test_sample_noisy_quality_one_matches_true(self):
        """High quality samples should roughly match the true distribution."""
        rng = np.random.default_rng(42)
        env = DataEnvironment(n_states=3, true_dist=np.array([0.7, 0.2, 0.1]), rng=rng)
        samples = env.sample_noisy(quality=1.0, n=10_000)
        counts = np.bincount(samples, minlength=3) / len(samples)
        np.testing.assert_allclose(counts, env.true_dist, atol=0.03)

    def test_sample_noisy_quality_zero_is_uniform(self):
        rng = np.random.default_rng(42)
        env = DataEnvironment(n_states=4, true_dist=np.array([1, 0, 0, 0], dtype=float), rng=rng)
        samples = env.sample_noisy(quality=0.0, n=10_000)
        counts = np.bincount(samples, minlength=4) / len(samples)
        np.testing.assert_allclose(counts, 0.25, atol=0.03)

    def test_kl_divergence_zero_for_true(self):
        env = DataEnvironment(n_states=5, rng=np.random.default_rng(0))
        assert env.kl_divergence(env.true_dist) < 1e-10

    def test_kl_divergence_positive_for_uniform(self):
        env = DataEnvironment(n_states=5, true_dist=np.array([0.8, 0.05, 0.05, 0.05, 0.05]))
        uniform = np.ones(5) / 5
        assert env.kl_divergence(uniform) > 0

    def test_optimal_decision_value(self):
        env = DataEnvironment(n_states=3, true_dist=np.array([0.1, 0.6, 0.3]))
        assert abs(env.optimal_decision_value() - 0.6) < 1e-10

    def test_decision_value_correct_choice(self):
        env = DataEnvironment(n_states=3, true_dist=np.array([0.1, 0.6, 0.3]))
        belief = np.array([0.0, 1.0, 0.0])  # correctly picks state 1
        assert abs(env.decision_value(belief) - 0.6) < 1e-10

    def test_decision_value_wrong_choice(self):
        env = DataEnvironment(n_states=3, true_dist=np.array([0.1, 0.6, 0.3]))
        belief = np.array([1.0, 0.0, 0.0])  # incorrectly picks state 0
        assert abs(env.decision_value(belief) - 0.1) < 1e-10
