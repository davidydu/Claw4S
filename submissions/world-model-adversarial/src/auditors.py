"""Auditors that evaluate manipulation patterns in signaling game results.

Each auditor examines a simulation trace and returns a dict of scalar metrics.

Auditors
--------
* **DistortionAuditor** -- measures KL divergence between learner beliefs
  and the true state distribution.
* **CredibilityExploitationAuditor** -- detects trust-then-exploit patterns
  by comparing distortion in early vs. late phases.
* **DecisionQualityAuditor** -- compares learner payoff to optimal.
* **RecoveryAuditor** -- measures how quickly beliefs recover after
  adversary switches to truthful signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass
class SimTrace:
    """Complete trace of one simulation run.

    Attributes
    ----------
    n_rounds : int
        Total number of rounds.
    true_states : NDArray[np.int64]
        True state at each round, shape (n_rounds,).
    signals : NDArray[np.int64]
        Signal sent at each round, shape (n_rounds,).
    actions : NDArray[np.int64]
        Learner's chosen action at each round, shape (n_rounds,).
    beliefs : NDArray[np.float64]
        Learner's belief vector at each round, shape (n_rounds, n_states).
    n_states : int
        Number of discrete states.
    """

    n_rounds: int
    true_states: NDArray[np.int64]
    signals: NDArray[np.int64]
    actions: NDArray[np.int64]
    beliefs: NDArray[np.float64]
    n_states: int


def _belief_error(beliefs: NDArray[np.float64], true_state: int) -> float:
    """1 - beliefs[true_state].  Ranges from 0 (perfect) to ~1 (worst)."""
    return float(1.0 - beliefs[true_state])


def _kl_from_true(beliefs: NDArray[np.float64], true_state: int, n_states: int) -> float:
    """KL(true || beliefs) with smoothing."""
    eps = 1e-12
    true_dist = np.zeros(n_states)
    true_dist[true_state] = 1.0
    p = true_dist + eps
    q = beliefs + eps
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


class DistortionAuditor:
    """Measures belief distortion across the simulation.

    Metrics
    -------
    mean_belief_error : float
        Mean (1 - belief[true_state]) across all rounds.
    final_belief_error : float
        Mean belief error in the last 20% of rounds.
    mean_kl : float
        Mean KL(true || beliefs) across all rounds.
    final_kl : float
        Mean KL in the last 20% of rounds.
    max_belief_error : float
        Maximum single-round belief error.
    """

    def audit(self, trace: SimTrace) -> dict[str, float]:
        errors = np.array([
            _belief_error(trace.beliefs[t], trace.true_states[t])
            for t in range(trace.n_rounds)
        ])
        kls = np.array([
            _kl_from_true(trace.beliefs[t], trace.true_states[t], trace.n_states)
            for t in range(trace.n_rounds)
        ])
        cutoff = int(0.8 * trace.n_rounds)
        return {
            "mean_belief_error": float(np.mean(errors)),
            "final_belief_error": float(np.mean(errors[cutoff:])),
            "mean_kl": float(np.mean(kls)),
            "final_kl": float(np.mean(kls[cutoff:])),
            "max_belief_error": float(np.max(errors)),
        }


class CredibilityExploitationAuditor:
    """Detects trust-then-exploit pattern.

    Compares signal truthfulness in the first ``n_early`` rounds
    vs. the last ``n_early`` rounds.  This captures the credibility
    phase (typically 200 rounds) without diluting it.

    Metrics
    -------
    early_truthful_rate : float
        Fraction of signals matching the true state in the first phase.
    late_truthful_rate : float
        Fraction of signals matching the true state in the second phase.
    exploitation_gap : float
        early_truthful_rate - late_truthful_rate.  Positive = credibility
        exploitation detected.
    early_belief_error : float
        Mean belief error in the first phase.
    late_belief_error : float
        Mean belief error in the second phase.
    """

    def __init__(self, n_early: int = 500):
        self.n_early = n_early

    def audit(self, trace: SimTrace) -> dict[str, float]:
        n_early = min(self.n_early, trace.n_rounds // 2)
        early_truthful = np.mean(
            trace.signals[:n_early] == trace.true_states[:n_early]
        )
        late_truthful = np.mean(
            trace.signals[-n_early:] == trace.true_states[-n_early:]
        )

        early_errors = np.array([
            _belief_error(trace.beliefs[t], trace.true_states[t])
            for t in range(n_early)
        ])
        late_errors = np.array([
            _belief_error(trace.beliefs[t], trace.true_states[t])
            for t in range(trace.n_rounds - n_early, trace.n_rounds)
        ])
        return {
            "early_truthful_rate": float(early_truthful),
            "late_truthful_rate": float(late_truthful),
            "exploitation_gap": float(early_truthful - late_truthful),
            "early_belief_error": float(np.mean(early_errors)),
            "late_belief_error": float(np.mean(late_errors)),
        }


class DecisionQualityAuditor:
    """Measures learner's decision quality relative to optimal.

    The learner's payoff for choosing action *a* when true state is *s*
    is 1 if a == s, else 0 (exact match).

    Metrics
    -------
    accuracy : float
        Fraction of rounds where action == true state.
    final_accuracy : float
        Accuracy in the last 20% of rounds.
    mean_payoff : float
        Same as accuracy (binary payoff).
    regret : float
        1 - accuracy (gap from optimal).
    """

    def audit(self, trace: SimTrace) -> dict[str, float]:
        correct = (trace.actions == trace.true_states).astype(float)
        cutoff = int(0.8 * trace.n_rounds)
        accuracy = float(np.mean(correct))
        final_accuracy = float(np.mean(correct[cutoff:]))
        return {
            "accuracy": accuracy,
            "final_accuracy": final_accuracy,
            "mean_payoff": accuracy,
            "regret": 1.0 - accuracy,
        }


class RecoveryAuditor:
    """Measures how quickly belief error drops in windows where
    signals become truthful.

    Scans the trace for windows where signal == true_state for at least
    ``window_size`` consecutive rounds, and measures how quickly the
    belief error drops during those windows.

    Metrics
    -------
    n_recovery_windows : int
        Number of truthful windows found.
    mean_recovery_rate : float
        Mean rate of belief error reduction per round during recovery
        windows.  Positive = beliefs are recovering.
    mean_recovery_start_error : float
        Mean belief error at the start of recovery windows.
    mean_recovery_end_error : float
        Mean belief error at the end of recovery windows.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size

    def audit(self, trace: SimTrace) -> dict[str, float]:
        truthful = trace.signals == trace.true_states
        # Find windows of consecutive truthful signals.
        windows: list[tuple[int, int]] = []
        start = None
        for t in range(trace.n_rounds):
            if truthful[t]:
                if start is None:
                    start = t
            else:
                if start is not None and (t - start) >= self.window_size:
                    windows.append((start, t))
                start = None
        if start is not None and (trace.n_rounds - start) >= self.window_size:
            windows.append((start, trace.n_rounds))

        if not windows:
            return {
                "n_recovery_windows": 0,
                "mean_recovery_rate": 0.0,
                "mean_recovery_start_error": 0.0,
                "mean_recovery_end_error": 0.0,
            }

        rates: list[float] = []
        start_errors: list[float] = []
        end_errors: list[float] = []
        for s, e in windows:
            err_s = _belief_error(trace.beliefs[s], trace.true_states[s])
            err_e = _belief_error(trace.beliefs[e - 1], trace.true_states[e - 1])
            length = e - s
            rate = (err_s - err_e) / length if length > 0 else 0.0
            rates.append(rate)
            start_errors.append(err_s)
            end_errors.append(err_e)

        return {
            "n_recovery_windows": len(windows),
            "mean_recovery_rate": float(np.mean(rates)),
            "mean_recovery_start_error": float(np.mean(start_errors)),
            "mean_recovery_end_error": float(np.mean(end_errors)),
        }


# ---------------------------------------------------------------------------
# Convenience: run all auditors
# ---------------------------------------------------------------------------

ALL_AUDITORS = {
    "distortion": DistortionAuditor(),
    "credibility": CredibilityExploitationAuditor(),
    "decision_quality": DecisionQualityAuditor(),
    "recovery": RecoveryAuditor(),
}


def run_all_auditors(trace: SimTrace) -> dict[str, dict[str, float]]:
    """Run every auditor on *trace* and return nested results."""
    return {name: auditor.audit(trace) for name, auditor in ALL_AUDITORS.items()}
