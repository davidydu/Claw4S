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
            evaluator_name="KuramotoOrder",
            sync_score=mean_r,
            evidence={"mean_r": mean_r, "window_steps": len(r_values)},
        )


# ---------------------------------------------------------------------------
# Evaluator 2: Spatial Alignment
# ---------------------------------------------------------------------------

class SpatialAlignmentEvaluator(BaseEvaluator):
    """Pearson correlation between spatial distance and circular phase distance.

    Intuition: well-synced ensembles show NO spatial ordering of phase, so
    the correlation should be near zero or negative.

    score = max(0, 1 - |r|)   where r is the Pearson correlation.

    For a perfectly synced ensemble all phase distances are 0, making the
    correlation undefined (zero variance).  In that edge case we return
    score = 1.0 (fully synced).

    Circular phase distance: min(|θ_i - θ_j|, 2π - |θ_i - θ_j|)
    """

    def evaluate(self, phase_history, positions, adjacency, sigma) -> EvalResult:
        window = self._final_slice(phase_history)          # (W, n)
        mean_phases = np.mean(window, axis=0)              # (n,) mean phase per dancer

        n = positions.shape[0]
        spatial_dists = []
        phase_dists = []

        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean spatial distance
                sd = float(np.linalg.norm(positions[i] - positions[j]))
                # Circular phase distance
                diff = abs(mean_phases[i] - mean_phases[j])
                pd = min(diff, 2 * np.pi - diff)
                spatial_dists.append(sd)
                phase_dists.append(pd)

        spatial_dists = np.array(spatial_dists)
        phase_dists = np.array(phase_dists)

        # If all phase distances are zero (perfect sync), correlation undefined
        if np.std(phase_dists) < 1e-12:
            pearson_r = 0.0
            score = 1.0
        else:
            # numpy corrcoef returns the full 2×2 matrix
            corr = np.corrcoef(spatial_dists, phase_dists)
            pearson_r = float(corr[0, 1])
            score = max(0.0, 1.0 - abs(pearson_r))

        return EvalResult(
            evaluator_name="SpatialAlignment",
            sync_score=score,
            evidence={"pearson_r": pearson_r, "n_pairs": len(spatial_dists)},
        )


# ---------------------------------------------------------------------------
# Evaluator 3: Velocity Synchrony
# ---------------------------------------------------------------------------

class VelocitySynchronyEvaluator(BaseEvaluator):
    """Variance of angular velocities normalised by σ².

    Angular velocity ≈ finite difference of phase over time.
    For a synced ensemble velocities should cluster tightly, giving low
    variance.  We normalise by σ² (noise power) to make it scale-free.

    score = max(0, 1 - var / σ²)

    Edge case: if σ == 0 and var == 0 → score = 1.0.
    """

    def evaluate(self, phase_history, positions, adjacency, sigma) -> EvalResult:
        window = self._final_slice(phase_history)          # (W, n)

        if window.shape[0] < 2:
            # Cannot compute velocities from a single frame
            return EvalResult(
                evaluator_name="VelocitySynchrony",
                sync_score=0.0,
                evidence={"velocity_variance": float("nan"), "sigma_sq": sigma ** 2},
            )

        velocities = np.diff(window, axis=0)               # (W-1, n)
        vel_variance = float(np.var(velocities))

        sigma_sq = sigma ** 2
        if sigma_sq < 1e-12:
            score = 1.0 if vel_variance < 1e-12 else 0.0
        else:
            score = max(0.0, 1.0 - vel_variance / sigma_sq)

        return EvalResult(
            evaluator_name="VelocitySynchrony",
            sync_score=score,
            evidence={"velocity_variance": vel_variance, "sigma_sq": sigma_sq},
        )


# ---------------------------------------------------------------------------
# Evaluator 4: Pairwise Entrainment
# ---------------------------------------------------------------------------

class PairwiseEntrainmentEvaluator(BaseEvaluator):
    """Fraction of connected pairs whose phase difference variance < 0.1.

    A pair (i, j) is "entrained" if the variance of their phase difference
    over the final 20% of the run is below 0.1 (radians²).
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
                # Wrap differences into [-π, π] for circular correctness
                diff = (diff + np.pi) % (2 * np.pi) - np.pi
                var = float(np.var(diff))
                total += 1
                if var < self.THRESHOLD:
                    entrained += 1

        score = (entrained / total) if total > 0 else 0.0

        return EvalResult(
            evaluator_name="PairwiseEntrainment",
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
        "KuramotoOrder": 0.35,
        "SpatialAlignment": 0.20,
        "VelocitySynchrony": 0.25,
        "PairwiseEntrainment": 0.20,
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
