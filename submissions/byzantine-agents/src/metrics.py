"""Metrics computation for Byzantine fault tolerance analysis.

Four metrics:
  1. Decision accuracy — fraction of correct committee decisions.
  2. Byzantine threshold — the exact fraction f* where accuracy drops below 50%.
  3. Byzantine amplification — ratio of accuracy degradation from
     strategic vs random Byzantine agents.
  4. Resilience score — area under the accuracy-vs-f curve (trapezoidal).
"""

from __future__ import annotations

import numpy as np
from typing import Sequence

from src.simulation import SimResult


def decision_accuracy(result: SimResult) -> float:
    """Return decision accuracy (already computed in SimResult)."""
    return result.accuracy


def byzantine_threshold(
    fractions: Sequence[float],
    accuracies: Sequence[float],
    cutoff: float = 0.50,
) -> float:
    """Estimate the Byzantine fraction f* where accuracy first drops below *cutoff*.

    Uses linear interpolation between the two surrounding data points.
    Returns 1.0 if accuracy never drops below the cutoff.
    Returns 0.0 if accuracy is below cutoff even at f=0 (broken baseline).
    """
    fracs = list(fractions)
    accs = list(accuracies)

    # If accuracy at f=0 is already below cutoff, threshold is 0
    if accs[0] < cutoff:
        return 0.0

    for i in range(1, len(fracs)):
        if accs[i] < cutoff:
            # Linear interpolation
            f_lo, f_hi = fracs[i - 1], fracs[i]
            a_lo, a_hi = accs[i - 1], accs[i]
            if a_lo == a_hi:
                return f_lo
            f_star = f_lo + (cutoff - a_lo) * (f_hi - f_lo) / (a_hi - a_lo)
            return float(f_star)

    return 1.0  # never dropped below cutoff


def byzantine_amplification(
    accuracy_at_f_strategic: float,
    accuracy_at_f_random: float,
    baseline_accuracy: float,
) -> float:
    """How much worse is strategic vs random Byzantine at the same fraction.

    Defined as:
        amplification = (baseline - strategic) / max(baseline - random, epsilon)

    A value > 1 means strategic Byzantine degrades accuracy more than random.
    """
    eps = 1e-9
    drop_strategic = baseline_accuracy - accuracy_at_f_strategic
    drop_random = baseline_accuracy - accuracy_at_f_random
    return float(drop_strategic / max(drop_random, eps))


def resilience_score(
    fractions: Sequence[float],
    accuracies: Sequence[float],
) -> float:
    """Area under the accuracy-vs-Byzantine-fraction curve.

    Uses the trapezoidal rule. Normalized to [0, 1] by dividing by
    the maximum possible area (accuracy=1.0 across all fractions).
    Higher is better — a perfectly resilient system scores 1.0.
    """
    trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    numerator = trapz(accuracies, fractions)
    denominator = max(trapz(np.ones_like(accuracies), fractions), 1e-9)
    return float(numerator / denominator)
