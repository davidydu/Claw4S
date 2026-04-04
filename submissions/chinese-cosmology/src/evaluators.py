# src/evaluators.py
"""Consistency evaluator panel for cross-system analysis.

Four evaluators measure agreement between BaZi, Zi Wei Dou Shu, and Wu Xing
domain scores across a collection of birth charts:

  CorrelationEvaluator        — Pearson |r|
  MutualInformationEvaluator  — normalized mutual information, 10-bin discretization
  DomainAgreementEvaluator    — fraction of charts where both systems agree on
                                 favorability (both > 0.5 or both ≤ 0.5)
  WuXingPredictivenessEvaluator — R² from linear regression of Wu Xing scores
                                   onto BaZi/ZiWei scores

EvaluatorPanel runs all four and returns a list of ConsistencyResult objects.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class ConsistencyResult:
    """Result from a single consistency evaluator.

    Attributes:
        evaluator_name    : human-readable name of the evaluator
        consistency_score : float in [0, 1]; higher = more consistent
        evidence          : dict with supporting statistics
    """
    evaluator_name: str
    consistency_score: float
    evidence: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEvaluator(ABC):
    """Abstract base class for consistency evaluators."""

    @abstractmethod
    def evaluate(self, scores_a: list, scores_b: list) -> ConsistencyResult:
        """Measure consistency between two parallel arrays of domain scores.

        Args:
            scores_a: list/array of float in [0, 1] — scores from system A
            scores_b: list/array of float in [0, 1] — scores from system B
                      (same length as scores_a)

        Returns:
            ConsistencyResult
        """


# ---------------------------------------------------------------------------
# Helper: pure-Python statistics (no scipy dependency)
# ---------------------------------------------------------------------------

def _mean(xs):
    n = len(xs)
    return sum(xs) / n if n > 0 else 0.0


def _pearson_r(xs, ys):
    """Compute Pearson r between two equal-length sequences.

    Returns 0.0 if either sequence has zero variance.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    mx, my = _mean(xs), _mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x < 1e-15 or denom_y < 1e-15:
        return 0.0
    return num / (denom_x * denom_y)


def _linear_regression_r2(xs, ys):
    """Compute R² for linear regression y ~ x.

    Returns 0.0 for degenerate cases.
    """
    n = len(xs)
    if n < 2:
        return 0.0
    r = _pearson_r(xs, ys)
    return r * r  # R² = r² for simple linear regression


# ---------------------------------------------------------------------------
# Evaluator 1: Pearson Correlation
# ---------------------------------------------------------------------------

class CorrelationEvaluator(BaseEvaluator):
    """Pearson |r| between two systems' domain scores.

    A perfect linear relationship → score = 1.0.
    Uncorrelated systems → score ≈ 0.
    """

    def evaluate(self, scores_a: list, scores_b: list) -> ConsistencyResult:
        r = _pearson_r(list(scores_a), list(scores_b))
        score = abs(r)
        return ConsistencyResult(
            evaluator_name="correlation",
            consistency_score=round(score, 8),
            evidence={"pearson_r": round(r, 8), "n": len(scores_a)},
        )


# ---------------------------------------------------------------------------
# Evaluator 2: Mutual Information
# ---------------------------------------------------------------------------

class MutualInformationEvaluator(BaseEvaluator):
    """Normalized mutual information between two systems (10-bin discretization).

    MI is normalized by max(H(A), H(B)) to give a score in [0, 1].
    """

    N_BINS: int = 10

    def evaluate(self, scores_a: list, scores_b: list) -> ConsistencyResult:
        n = len(scores_a)
        if n < 2:
            return ConsistencyResult(
                evaluator_name="mutual_information",
                consistency_score=0.0,
                evidence={"mi": 0.0, "n": n},
            )

        # Discretize into bins
        def digitize(vals):
            bins = [i / self.N_BINS for i in range(1, self.N_BINS)]
            result = []
            for v in vals:
                b = 0
                for threshold in bins:
                    if v > threshold:
                        b += 1
                result.append(b)
            return result

        a_bins = digitize(list(scores_a))
        b_bins = digitize(list(scores_b))

        # Count joint and marginal frequencies
        joint = {}
        marg_a = {}
        marg_b = {}
        for a, b in zip(a_bins, b_bins):
            joint[(a, b)] = joint.get((a, b), 0) + 1
            marg_a[a] = marg_a.get(a, 0) + 1
            marg_b[b] = marg_b.get(b, 0) + 1

        # Compute MI, H(A), H(B)
        mi = 0.0
        for (a, b), cnt in joint.items():
            p_ab = cnt / n
            p_a = marg_a[a] / n
            p_b = marg_b[b] / n
            if p_ab > 0 and p_a > 0 and p_b > 0:
                mi += p_ab * math.log(p_ab / (p_a * p_b))

        h_a = -sum((c / n) * math.log(c / n) for c in marg_a.values() if c > 0)
        h_b = -sum((c / n) * math.log(c / n) for c in marg_b.values() if c > 0)

        norm_denom = max(h_a, h_b)
        nmi = (mi / norm_denom) if norm_denom > 1e-12 else 0.0
        nmi = max(0.0, min(nmi, 1.0))

        return ConsistencyResult(
            evaluator_name="mutual_information",
            consistency_score=round(nmi, 8),
            evidence={
                "mi_nats": round(mi, 8),
                "h_a": round(h_a, 8),
                "h_b": round(h_b, 8),
                "n": n,
            },
        )


# ---------------------------------------------------------------------------
# Evaluator 3: Domain Agreement
# ---------------------------------------------------------------------------

class DomainAgreementEvaluator(BaseEvaluator):
    """Fraction of charts where systems agree on favorability.

    Agreement: both > 0.5 (favorable) or both ≤ 0.5 (unfavorable).
    """

    def evaluate(self, scores_a: list, scores_b: list) -> ConsistencyResult:
        n = len(scores_a)
        if n == 0:
            return ConsistencyResult(
                evaluator_name="domain_agreement",
                consistency_score=0.0,
                evidence={"n": 0},
            )

        agree = sum(
            1
            for a, b in zip(scores_a, scores_b)
            if (a > 0.5 and b > 0.5) or (a <= 0.5 and b <= 0.5)
        )
        score = agree / n

        return ConsistencyResult(
            evaluator_name="domain_agreement",
            consistency_score=round(score, 8),
            evidence={"agreements": agree, "n": n},
        )


# ---------------------------------------------------------------------------
# Evaluator 4: Wu Xing Predictiveness
# ---------------------------------------------------------------------------

class WuXingPredictivenessEvaluator(BaseEvaluator):
    """R² from linear regression of Wu Xing scores onto target scores.

    Measures how well Wu Xing element dynamics predict BaZi or Zi Wei ratings.
    Score is R² = r² from simple linear regression, clamped to [0, 1].
    """

    def evaluate(self, scores_a: list, scores_b: list) -> ConsistencyResult:
        """
        Args:
            scores_a: Wu Xing domain scores (predictor)
            scores_b: BaZi or Zi Wei domain scores (target)
        """
        r2 = _linear_regression_r2(list(scores_a), list(scores_b))
        r2 = max(0.0, min(r2, 1.0))

        return ConsistencyResult(
            evaluator_name="wuxing_predictiveness",
            consistency_score=round(r2, 8),
            evidence={"r_squared": round(r2, 8), "n": len(scores_a)},
        )


# ---------------------------------------------------------------------------
# Evaluator Panel
# ---------------------------------------------------------------------------

class EvaluatorPanel:
    """Runs all four consistency evaluators and returns results.

    Usage:
        panel = EvaluatorPanel()
        results = panel.evaluate_all(bazi_scores, ziwei_scores, wuxing_scores)
        # results: list of 4 ConsistencyResult objects
    """

    def __init__(self):
        self._evaluators: List[BaseEvaluator] = [
            CorrelationEvaluator(),
            MutualInformationEvaluator(),
            DomainAgreementEvaluator(),
            WuXingPredictivenessEvaluator(),
        ]

    def evaluate_all(
        self,
        bazi_scores: list,
        ziwei_scores: list,
        wuxing_scores: list,
    ) -> List[ConsistencyResult]:
        """Run all evaluators on BaZi vs Zi Wei scores.

        Evaluator 1-3 compare bazi_scores vs ziwei_scores.
        Evaluator 4 uses wuxing_scores as predictor for ziwei_scores.

        Args:
            bazi_scores:   list of floats in [0, 1]
            ziwei_scores:  list of floats in [0, 1], same length as bazi_scores
            wuxing_scores: list of floats in [0, 1], same length as bazi_scores

        Returns:
            list of 4 ConsistencyResult objects (one per evaluator)
        """
        results = []
        for ev in self._evaluators:
            if isinstance(ev, WuXingPredictivenessEvaluator):
                # Wu Xing predicts Zi Wei
                results.append(ev.evaluate(wuxing_scores, ziwei_scores))
            else:
                results.append(ev.evaluate(bazi_scores, ziwei_scores))
        return results
