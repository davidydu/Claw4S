"""Tests for learner and adversary agents."""

import numpy as np
import pytest

from src.agents import (
    NaiveLearner,
    SkepticalLearner,
    AdaptiveLearner,
    RandomAdversary,
    StrategicAdversary,
    PatientAdversary,
    make_learner,
    make_adversary,
)


# ---------------------------------------------------------------------------
# Learner tests
# ---------------------------------------------------------------------------

class TestNaiveLearner:
    def test_beliefs_start_uniform(self):
        nl = NaiveLearner(5)
        np.testing.assert_allclose(nl.beliefs, np.ones(5) / 5)

    def test_single_signal_shifts_belief(self):
        nl = NaiveLearner(5)
        nl.update(2)
        assert nl.beliefs[2] > 0.2  # Should be higher than uniform.
        assert nl.beliefs.sum() == pytest.approx(1.0)

    def test_repeated_signals_concentrate(self):
        nl = NaiveLearner(5)
        for _ in range(20):
            nl.update(3)
        assert nl.beliefs[3] > 0.9

    def test_belief_floor_prevents_zero(self):
        nl = NaiveLearner(5, belief_floor=0.01)
        for _ in range(100):
            nl.update(0)
        # Even after 100 signals for state 0, other states retain some mass.
        assert nl.beliefs[1] > 0
        assert nl.beliefs[2] > 0

    def test_reset(self):
        nl = NaiveLearner(5)
        nl.update(0)
        nl.reset()
        np.testing.assert_allclose(nl.beliefs, np.ones(5) / 5)


class TestSkepticalLearner:
    def test_trust_zero_ignores_signals(self):
        sl = SkepticalLearner(5, trust=0.0)
        sl.update(2)
        # With trust=0, likelihood is uniform => beliefs stay uniform.
        np.testing.assert_allclose(sl.beliefs, np.ones(5) / 5, atol=1e-10)

    def test_trust_one_equals_naive(self):
        nl = NaiveLearner(5, signal_strength=3.0, belief_floor=0.0)
        sl = SkepticalLearner(5, trust=1.0, signal_strength=3.0, belief_floor=0.0)
        nl.update(2)
        sl.update(2)
        np.testing.assert_allclose(sl.beliefs, nl.beliefs, atol=1e-12)

    def test_partial_trust_less_concentrated(self):
        nl = NaiveLearner(5, signal_strength=3.0)
        sl = SkepticalLearner(5, trust=0.4, signal_strength=3.0)
        for _ in range(10):
            nl.update(1)
            sl.update(1)
        # Skeptical learner should be less concentrated on state 1.
        assert sl.beliefs[1] < nl.beliefs[1]


class TestAdaptiveLearner:
    def test_trust_decreases_under_random_noise(self):
        al = AdaptiveLearner(5, initial_trust=0.7, ema_alpha=0.1)
        rng = np.random.default_rng(42)
        # Send uniformly random signals.  The learner's action will
        # drift around and only occasionally match the previous signal,
        # so measured accuracy is low and trust drops.
        for _ in range(200):
            al.update(int(rng.integers(0, 5)))
            al.choose_action()
        assert al.trust < 0.5

    def test_trust_stays_high_under_truthful_signals(self):
        al = AdaptiveLearner(5, initial_trust=0.7, ema_alpha=0.05)
        # Consistently signal state 2.
        for _ in range(200):
            al.update(2)
            al.choose_action()
        # Since signals always say 2 and learner converges to 2,
        # accuracy is 1.0 => trust should rise.
        assert al.trust > 0.7


# ---------------------------------------------------------------------------
# Adversary tests
# ---------------------------------------------------------------------------

class TestRandomAdversary:
    def test_signals_in_range(self):
        ra = RandomAdversary(5, rng=np.random.default_rng(0))
        beliefs = np.ones(5) / 5
        for _ in range(100):
            s = ra.choose_signal(0, beliefs)
            assert 0 <= s < 5

    def test_signals_are_diverse(self):
        ra = RandomAdversary(5, rng=np.random.default_rng(0))
        beliefs = np.ones(5) / 5
        signals = {ra.choose_signal(0, beliefs) for _ in range(200)}
        assert len(signals) > 1


class TestStrategicAdversary:
    def test_never_sends_true_state(self):
        sa = StrategicAdversary(5, rng=np.random.default_rng(0))
        beliefs = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        for true_state in range(5):
            signal = sa.choose_signal(true_state, beliefs)
            assert signal != true_state

    def test_picks_highest_non_true_belief(self):
        sa = StrategicAdversary(5)
        beliefs = np.array([0.05, 0.6, 0.2, 0.1, 0.05])
        # True state = 1, so should pick state 2 (next highest).
        assert sa.choose_signal(1, beliefs) == 2


class TestPatientAdversary:
    def test_truthful_during_credibility_phase(self):
        pa = PatientAdversary(5, credibility_rounds=50)
        beliefs = np.ones(5) / 5
        for _ in range(50):
            signal = pa.choose_signal(3, beliefs)
            assert signal == 3  # Truthful.

    def test_deceptive_after_credibility_phase(self):
        pa = PatientAdversary(5, credibility_rounds=10)
        beliefs = np.array([0.3, 0.3, 0.2, 0.1, 0.1])
        # Burn through credibility phase.
        for _ in range(10):
            pa.choose_signal(0, beliefs)
        # Now should be deceptive.
        signal = pa.choose_signal(0, beliefs)
        assert signal != 0

    def test_reset_resets_round_counter(self):
        pa = PatientAdversary(5, credibility_rounds=5)
        beliefs = np.ones(5) / 5
        for _ in range(10):
            pa.choose_signal(0, beliefs)
        pa.reset()
        # After reset, should be truthful again.
        assert pa.choose_signal(2, beliefs) == 2


# ---------------------------------------------------------------------------
# Factory tests
# ---------------------------------------------------------------------------

class TestFactories:
    @pytest.mark.parametrize("code", ["NL", "SL", "AL"])
    def test_make_learner(self, code):
        learner = make_learner(code, 5)
        assert learner.n_states == 5

    @pytest.mark.parametrize("code", ["RA", "SA", "PA"])
    def test_make_adversary(self, code):
        a = make_adversary(code, 5)
        assert a.n_states == 5
