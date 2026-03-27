"""Sync Evaluator Panel (Layer 3).

Four independent evaluators judge whether a dancer ensemble has achieved
synchrony, then an EvaluatorPanel aggregates their verdicts.

All evaluators operate on the **final 20%** of the phase history.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result produced by a single evaluator."""
    evaluator_name: str
    sync_score: float          # 0.0 (no sync) to 1.0 (perfect sync)
    evidence: Dict             # evaluator-specific diagnostic data


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEvaluator(abc.ABC):
    """All evaluators share the same interface."""

    @abc.abstractmethod
    def evaluate(
        self,
        phase_history: np.ndarray,   # shape (T, n) — full simulation history
        positions: np.ndarray,        # shape (n, 2) — spatial XY coordinates
        adjacency: Dict[int, List[int]],  # connectivity graph
        sigma: float,                 # noise standard deviation used in sim
    ) -> EvalResult:
        """Return an EvalResult for the given simulation run."""

    # ------------------------------------------------------------------
    # Shared helper
    # ------------------------------------------------------------------

    @staticmethod
    def _final_slice(phase_history: np.ndarray) -> np.ndarray:
        """Return the final 20% of time steps."""
        T = phase_history.shape[0]
        start = max(0, int(0.8 * T))
        return phase_history[start:]


# ---------------------------------------------------------------------------
# Evaluator 1: Kuramoto Order Parameter
# ---------------------------------------------------------------------------

class KuramotoOrderEvaluator(BaseEvaluator):
    """Mean Kuramoto order parameter r over the final 20% of history.

    r = |mean(exp(i * theta))| ranges from 0 (incoherent) to 1 (fully synced).
    """

    def evaluate(self, phase_history, positions, adjacency, sigma) -> EvalResult:
        window = self._final_slice(phase_history)          # (W, n)
        r_values = np.abs(np.mean(np.exp(1j * window), axis=1))  # (W,)
        mean_r = float(np.mean(r_values))
        return EvalResult(
            evaluator_name="kuramoto_order",
            sync_score=mean_r,
            evidence={"mean_r": mean_r, "window_steps": len(r_values)},
        )


# ---------------------------------------------------------------------------
# Evaluator 2: Spatial Alignment
# ---------------------------------------------------------------------------

class SpatialAlignmentEvaluator(BaseEvaluator):
    """Mean pairwise circular phase distance, mapped to a [0,1] sync score.

    Computes the mean circular phase distance across all n*(n-1)/2 pairs
    using the time-averaged phase per dancer in the final 20% of timesteps.
    Maps to score via: score = exp(-mean_phase_spread / pi).

    score = 1.0 when all phases are identical (mean_phase_spread = 0).
    score -> exp(-1) ~ 0.368 for random incoherent phases (spread ~ pi/2).

    Uses vectorized numpy for speed.
    """

    def evaluate(self, phase_history, positions, adjacency, sigma) -> EvalResult:
        window = self._final_slice(phase_history)          # (W, n)
        # Circular mean phase per dancer
        mean_phases = np.angle(np.mean(np.exp(1j * window), axis=0))  # (n,)

        n = len(mean_phases)
        if n < 2:
            return EvalResult(
                evaluator_name="spatial_alignment",
                sync_score=1.0,
                evidence={"mean_phase_spread": 0.0, "n_pairs": 0},
            )

        # Vectorized pairwise circular phase distances
        # diff[i, j] = mean_phases[i] - mean_phases[j]
        diff = mean_phases[:, None] - mean_phases[None, :]  # (n, n)
        # Circular wrap to [0, pi]
        circ_dist = np.abs((diff + np.pi) % (2 * np.pi) - np.pi)  # (n, n)
        # Take upper triangle only
        i_idx, j_idx = np.triu_indices(n, k=1)
        phase_dists = circ_dist[i_idx, j_idx]

        mean_phase_spread = float(np.mean(phase_dists))
        score = float(np.exp(-mean_phase_spread / np.pi))

        return EvalResult(
            evaluator_name="spatial_alignment",
            sync_score=score,
            evidence={"mean_phase_spread": mean_phase_spread, "n_pairs": len(phase_dists)},
        )


# ---------------------------------------------------------------------------
# Evaluator 3: Velocity Synchrony
# ---------------------------------------------------------------------------

class VelocitySynchronyEvaluator(BaseEvaluator):
    """Pairwise phase-difference variance, mapped to a [0,1] sync score.

    For a frequency-locked pair (i, j), the phase difference theta_i - theta_j
    should be approximately constant over time. We measure the MEAN pairwise
    phase-difference variance across ALL n*(n-1)/2 pairs in the final 20%
    of history, then map to [0,1]:

        score = exp(-mean_pairwise_var / threshold)

    where threshold = 0.5 (radians^2). score ~1 when locked, ~0 when not.

    Uses vectorized numpy for speed (no Python loops over pairs).
    """

    THRESHOLD = 0.5  # radians^2 scale for score normalization

    def evaluate(self, phase_history, positions, adjacency, sigma) -> EvalResult:
        window = self._final_slice(phase_history)          # (W, n)

        if window.shape[0] < 2:
            return EvalResult(
                evaluator_name="velocity_synchrony",
                sync_score=0.0,
                evidence={"mean_pairwise_var": float("nan"), "n_pairs": 0},
            )

        n = window.shape[1]
        if n < 2:
            return EvalResult(
                evaluator_name="velocity_synchrony",
                sync_score=1.0,
                evidence={"mean_pairwise_var": 0.0, "n_pairs": 0},
            )

        # Vectorized: diff_matrix[t, i, j] = window[t, i] - window[t, j]
        # But that's O(T * n^2) memory. Instead compute per-pair variance efficiently.
        # diff[t, pair] for all upper-triangle pairs
        i_idx, j_idx = np.triu_indices(n, k=1)  # (n_pairs,)
        # diff_series shape: (W, n_pairs)
        diff_series = window[:, i_idx] - window[:, j_idx]  # (W, n_pairs)
        # Wrap to [-pi, pi]
        diff_series = (diff_series + np.pi) % (2 * np.pi) - np.pi
        # Variance across time for each pair
        pair_vars = np.var(diff_series, axis=0)  # (n_pairs,)

        mean_pairwise_var = float(np.mean(pair_vars))
        n_pairs = len(pair_vars)
        score = float(np.exp(-mean_pairwise_var / self.THRESHOLD))

        return EvalResult(
            evaluator_name="velocity_synchrony",
            sync_score=score,
            evidence={"mean_pairwise_var": mean_pairwise_var, "n_pairs": n_pairs},
        )


# ---------------------------------------------------------------------------
# Evaluator 4: Pairwise Entrainment
# ---------------------------------------------------------------------------

class PairwiseEntrainmentEvaluator(BaseEvaluator):
    """Fraction of connected pairs whose phase difference variance < 0.1.

    A pair (i, j) is "entrained" if the variance of their phase difference
    over the final 20% of the run is below 0.1 (radians^2).
    """

    THRESHOLD = 0.1

    def evaluate(self, phase_history, positions, adjacency, sigma) -> EvalResult:
        window = self._final_slice(phase_history)          # (W, n)

        entrained = 0
        total = 0

        for i, neighbours in adjacency.items():
            for j in neighbours:
                if j <= i:
                    continue  # count each pair once
                diff = window[:, i] - window[:, j]
                # Wrap differences into [-pi, pi] for circular correctness
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                var = float(np.var(diff))
                total += 1
                if var < self.THRESHOLD:
                    entrained += 1

        score = (entrained / total) if total > 0 else 0.0

        return EvalResult(
            evaluator_name="pairwise_entrainment",
            sync_score=score,
            evidence={"entrained_pairs": entrained, "total_pairs": total,
                      "threshold": self.THRESHOLD},
        )


# ---------------------------------------------------------------------------
# Evaluator Panel
# ---------------------------------------------------------------------------

# Default threshold for calling a run "synced"
_SYNC_THRESHOLD = 0.5


class EvaluatorPanel:
    """Runs all four evaluators and aggregates their verdicts.

    Aggregation modes
    -----------------
    majority    : synced if > 50% of evaluators score >= threshold
    unanimous   : synced if ALL evaluators score >= threshold
    weighted    : synced if weighted-average score >= threshold
    """

    # Weights for the "weighted" aggregation (must sum to 1)
    _WEIGHTS = {
        "kuramoto_order": 0.35,
        "spatial_alignment": 0.20,
        "velocity_synchrony": 0.25,
        "pairwise_entrainment": 0.20,
    }

    def __init__(self) -> None:
        self._evaluators: List[BaseEvaluator] = [
            KuramotoOrderEvaluator(),
            SpatialAlignmentEvaluator(),
            VelocitySynchronyEvaluator(),
            PairwiseEntrainmentEvaluator(),
        ]

    def evaluate_all(
        self,
        phase_history: np.ndarray,
        positions: np.ndarray,
        adjacency: Dict[int, List[int]],
        sigma: float,
    ) -> List[EvalResult]:
        """Run every evaluator and return the list of results."""
        return [
            ev.evaluate(phase_history, positions, adjacency, sigma)
            for ev in self._evaluators
        ]

    def aggregate(
        self,
        results: List[EvalResult],
        mode: str = "majority",
        threshold: float = _SYNC_THRESHOLD,
    ) -> bool:
        """Aggregate individual results into a single boolean verdict.

        Parameters
        ----------
        results   : output of evaluate_all()
        mode      : "majority" | "unanimous" | "weighted"
        threshold : score cutoff for "synced" (default 0.5)

        Returns
        -------
        True if the ensemble is judged synced, False otherwise.
        """
        if mode == "majority":
            votes = [r.sync_score >= threshold for r in results]
            return sum(votes) > len(votes) / 2

        elif mode == "unanimous":
            return all(r.sync_score >= threshold for r in results)

        elif mode == "weighted":
            weighted_score = sum(
                self._WEIGHTS.get(r.evaluator_name, 1.0 / len(results)) * r.sync_score
                for r in results
            )
            return weighted_score >= threshold

        else:
            raise ValueError(f"Unknown aggregation mode: {mode!r}")
