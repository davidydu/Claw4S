"""Tests for auditors."""

import numpy as np
import pytest

from src.auditors import (
    SimTrace,
    DistortionAuditor,
    CredibilityExploitationAuditor,
    DecisionQualityAuditor,
    RecoveryAuditor,
    run_all_auditors,
)


def _make_trace(
    n_rounds: int = 100,
    n_states: int = 5,
    true_states: np.ndarray | None = None,
    signals: np.ndarray | None = None,
    actions: np.ndarray | None = None,
    beliefs: np.ndarray | None = None,
) -> SimTrace:
    """Helper to construct a SimTrace with sensible defaults."""
    if true_states is None:
        true_states = np.zeros(n_rounds, dtype=np.int64)
    if signals is None:
        signals = np.zeros(n_rounds, dtype=np.int64)
    if actions is None:
        actions = np.zeros(n_rounds, dtype=np.int64)
    if beliefs is None:
        beliefs = np.full((n_rounds, n_states), 1.0 / n_states)
    return SimTrace(
        n_rounds=n_rounds,
        true_states=true_states,
        signals=signals,
        actions=actions,
        beliefs=beliefs,
        n_states=n_states,
    )


class TestDistortionAuditor:
    def test_perfect_beliefs_zero_error(self):
        n = 100
        beliefs = np.zeros((n, 5))
        true_states = np.zeros(n, dtype=np.int64)
        beliefs[:, 0] = 1.0  # Perfect beliefs for true state 0.
        trace = _make_trace(beliefs=beliefs, true_states=true_states)
        result = DistortionAuditor().audit(trace)
        assert result["mean_belief_error"] == pytest.approx(0.0, abs=1e-10)
        assert result["final_belief_error"] == pytest.approx(0.0, abs=1e-10)

    def test_uniform_beliefs_positive_error(self):
        trace = _make_trace()  # Uniform beliefs, true state 0.
        result = DistortionAuditor().audit(trace)
        # Uniform beliefs: error = 1 - 1/5 = 0.8
        assert result["mean_belief_error"] == pytest.approx(0.8, abs=1e-10)

    def test_wrong_beliefs_high_error(self):
        n = 100
        beliefs = np.zeros((n, 5))
        beliefs[:, 3] = 0.98
        beliefs[:, 0] = 0.005
        beliefs[:, 1] = 0.005
        beliefs[:, 2] = 0.005
        beliefs[:, 4] = 0.005
        true_states = np.zeros(n, dtype=np.int64)  # True state = 0.
        trace = _make_trace(beliefs=beliefs, true_states=true_states)
        result = DistortionAuditor().audit(trace)
        assert result["mean_belief_error"] > 0.9


class TestCredibilityExploitationAuditor:
    def test_uniform_truthfulness(self):
        """No exploitation pattern when signals are always truthful."""
        n = 100
        signals = np.zeros(n, dtype=np.int64)
        true_states = np.zeros(n, dtype=np.int64)
        trace = _make_trace(signals=signals, true_states=true_states)
        result = CredibilityExploitationAuditor(n_early=30).audit(trace)
        assert result["exploitation_gap"] == pytest.approx(0.0, abs=1e-10)
        assert result["early_truthful_rate"] == pytest.approx(1.0)

    def test_exploitation_pattern_detected(self):
        """Early truthful, late deceptive => positive exploitation gap."""
        n = 200
        true_states = np.zeros(n, dtype=np.int64)
        signals = np.zeros(n, dtype=np.int64)
        # First 50 rounds: truthful (signal = 0 = true state).
        # Remaining rounds: deceptive (signal = 3 != true state 0).
        signals[50:] = 3
        trace = _make_trace(n_rounds=n, signals=signals, true_states=true_states)
        result = CredibilityExploitationAuditor(n_early=50).audit(trace)
        assert result["early_truthful_rate"] == pytest.approx(1.0)
        assert result["late_truthful_rate"] == pytest.approx(0.0)
        assert result["exploitation_gap"] == pytest.approx(1.0)


class TestDecisionQualityAuditor:
    def test_perfect_accuracy(self):
        n = 100
        true_states = np.zeros(n, dtype=np.int64)
        actions = np.zeros(n, dtype=np.int64)
        trace = _make_trace(true_states=true_states, actions=actions)
        result = DecisionQualityAuditor().audit(trace)
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["regret"] == pytest.approx(0.0)

    def test_zero_accuracy(self):
        n = 100
        true_states = np.zeros(n, dtype=np.int64)
        actions = np.ones(n, dtype=np.int64)  # Always wrong.
        trace = _make_trace(true_states=true_states, actions=actions)
        result = DecisionQualityAuditor().audit(trace)
        assert result["accuracy"] == pytest.approx(0.0)
        assert result["regret"] == pytest.approx(1.0)

    def test_partial_accuracy(self):
        n = 100
        true_states = np.zeros(n, dtype=np.int64)
        actions = np.zeros(n, dtype=np.int64)
        actions[50:] = 1  # Wrong for last 50.
        trace = _make_trace(true_states=true_states, actions=actions)
        result = DecisionQualityAuditor().audit(trace)
        assert result["accuracy"] == pytest.approx(0.5)


class TestRecoveryAuditor:
    def test_no_truthful_windows(self):
        """All signals deceptive => no recovery windows."""
        n = 100
        true_states = np.zeros(n, dtype=np.int64)
        signals = np.ones(n, dtype=np.int64)  # Always lying.
        trace = _make_trace(true_states=true_states, signals=signals)
        result = RecoveryAuditor(window_size=10).audit(trace)
        assert result["n_recovery_windows"] == 0

    def test_truthful_window_detected(self):
        """A long truthful stretch should be detected."""
        n = 200
        true_states = np.zeros(n, dtype=np.int64)
        signals = np.ones(n, dtype=np.int64)
        # Truthful window from round 50 to 150.
        signals[50:150] = 0
        trace = _make_trace(n_rounds=n, true_states=true_states, signals=signals)
        result = RecoveryAuditor(window_size=50).audit(trace)
        assert result["n_recovery_windows"] >= 1

    def test_recovery_with_improving_beliefs(self):
        """Beliefs improve during truthful window => positive recovery rate."""
        n = 200
        n_states = 5
        true_states = np.zeros(n, dtype=np.int64)
        signals = np.ones(n, dtype=np.int64)
        signals[50:150] = 0  # Truthful window.
        # Build beliefs that improve during the truthful window.
        beliefs = np.full((n, n_states), 1.0 / n_states)
        # Before window: wrong beliefs (concentrated on state 3).
        for t in range(50):
            beliefs[t] = [0.01, 0.01, 0.01, 0.96, 0.01]
        # During window: gradually correct.
        for i, t in enumerate(range(50, 150)):
            frac = i / 100.0
            beliefs[t, 0] = 0.01 + frac * 0.95
            beliefs[t, 3] = 0.96 - frac * 0.95
            remaining = (1.0 - beliefs[t, 0] - beliefs[t, 3]) / 3
            beliefs[t, 1] = remaining
            beliefs[t, 2] = remaining
            beliefs[t, 4] = remaining
        trace = _make_trace(
            n_rounds=n, n_states=n_states,
            true_states=true_states, signals=signals, beliefs=beliefs,
        )
        result = RecoveryAuditor(window_size=50).audit(trace)
        assert result["mean_recovery_rate"] > 0
        assert result["mean_recovery_start_error"] > result["mean_recovery_end_error"]


class TestRunAllAuditors:
    def test_returns_all_sections(self):
        trace = _make_trace()
        results = run_all_auditors(trace)
        assert "distortion" in results
        assert "credibility" in results
        assert "decision_quality" in results
        assert "recovery" in results
        # Check that each section has metrics.
        assert "mean_belief_error" in results["distortion"]
        assert "accuracy" in results["decision_quality"]
