"""Tests for the HiddenEnvironment class."""

import numpy as np
import pytest

from src.environment import HiddenEnvironment


class TestHiddenEnvironment:
    """Basic construction and state management."""

    def test_initial_state_in_range(self):
        env = HiddenEnvironment(n_states=5, seed=42)
        assert 0 <= env.true_state < 5

    def test_true_state_distribution_is_one_hot(self):
        env = HiddenEnvironment(n_states=5, seed=0)
        dist = env.true_state_distribution()
        assert dist.sum() == pytest.approx(1.0)
        assert dist[env.true_state] == pytest.approx(1.0)

    def test_stable_never_drifts(self):
        env = HiddenEnvironment(n_states=5, drift_regime="stable", seed=7)
        initial = env.true_state
        for _ in range(10_000):
            env.step()
        assert env.true_state == initial

    def test_volatile_drifts(self):
        env = HiddenEnvironment(
            n_states=5, drift_regime="volatile", drift_interval=10, seed=99
        )
        states_seen: set[int] = {env.true_state}
        for _ in range(200):
            env.step()
            states_seen.add(env.true_state)
        # With 200 rounds and drift every 10, expect multiple states.
        assert len(states_seen) > 1

    def test_slow_drift_changes_at_interval(self):
        env = HiddenEnvironment(
            n_states=5, drift_regime="slow_drift", drift_interval=50, seed=3
        )
        initial = env.true_state
        # Step 49 times -- should still be same.
        for _ in range(49):
            env.step()
        assert env.true_state == initial
        # Step once more (round 50) -- may change.
        env.step()
        # We can't guarantee a change (could land on the same state),
        # so just check that the mechanism doesn't crash.

    def test_reset(self):
        env = HiddenEnvironment(n_states=5, seed=42)
        s0 = env.true_state
        for _ in range(100):
            env.step()
        env.reset(seed=42)
        assert env.true_state == s0
        assert env._round == 0


class TestNoisySignal:
    """Tests for generate_noisy_signal."""

    def test_zero_noise_is_identity(self):
        env = HiddenEnvironment(n_states=5, seed=0)
        for s in range(5):
            assert env.generate_noisy_signal(s, 0.0) == s

    def test_full_noise_is_random(self):
        env = HiddenEnvironment(n_states=5, seed=0)
        signals = {env.generate_noisy_signal(0, 1.0) for _ in range(500)}
        # With full noise we should see multiple distinct signals.
        assert len(signals) > 1

    def test_partial_noise_in_range(self):
        env = HiddenEnvironment(n_states=5, seed=0)
        for _ in range(100):
            sig = env.generate_noisy_signal(2, 0.3)
            assert 0 <= sig < 5
