"""Metrics for information cascade analysis.

Computes aggregate statistics from simulation results:
- Cascade formation rate: fraction of simulations where a cascade formed.
- Cascade accuracy: fraction of formed cascades that match the true state.
- Cascade fragility: fraction of cascades that are broken (cascade_length < n - start).
- Mean cascade length: average consecutive agents in a cascade.
"""

from __future__ import annotations
import math


def cascade_formation_rate(results: list[dict]) -> float:
    """Fraction of simulations where a cascade formed."""
    if not results:
        return 0.0
    n_formed = sum(1 for r in results if r["cascade_formed"])
    return n_formed / len(results)


def cascade_accuracy(results: list[dict]) -> float | None:
    """Fraction of formed cascades that matched the true state.

    Returns None if no cascades formed.
    """
    cascades = [r for r in results if r["cascade_formed"]]
    if not cascades:
        return None
    n_correct = sum(1 for r in cascades if r["cascade_correct"])
    return n_correct / len(cascades)


def cascade_fragility(results: list[dict]) -> float | None:
    """Fraction of formed cascades that were broken before the sequence ended.

    A cascade is "broken" if cascade_length < (n_agents - cascade_start).
    Returns None if no cascades formed.
    """
    cascades = [r for r in results if r["cascade_formed"]]
    if not cascades:
        return None
    n_broken = 0
    for r in cascades:
        remaining = r["n_agents"] - r["cascade_start"]
        if r["cascade_length"] < remaining:
            n_broken += 1
    return n_broken / len(cascades)


def mean_cascade_length(results: list[dict]) -> float | None:
    """Mean cascade length among simulations where a cascade formed.

    Returns None if no cascades formed.
    """
    cascades = [r for r in results if r["cascade_formed"]]
    if not cascades:
        return None
    total = sum(r["cascade_length"] for r in cascades)
    return total / len(cascades)


def majority_accuracy(results: list[dict]) -> float:
    """Fraction of simulations where the final majority action was correct."""
    if not results:
        return 0.0
    n_correct = sum(1 for r in results if r["majority_correct"])
    return n_correct / len(results)


def compute_group_metrics(results: list[dict]) -> dict:
    """Compute all metrics for a group of simulation results.

    Args:
        results: List of simulation result dicts (from run_single_simulation).

    Returns:
        Dict with all computed metrics.
    """
    return {
        "n_simulations": len(results),
        "cascade_formation_rate": cascade_formation_rate(results),
        "cascade_accuracy": cascade_accuracy(results),
        "cascade_fragility": cascade_fragility(results),
        "mean_cascade_length": mean_cascade_length(results),
        "majority_accuracy": majority_accuracy(results),
    }


def compute_standard_error(values: list[float]) -> float:
    """Compute standard error of the mean for a list of values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance / n)


def proportion_ci_95(count: int, total: int) -> tuple[float, float]:
    """Wilson score 95% confidence interval for a proportion.

    Returns (lower, upper) bounds.
    """
    if total == 0:
        return (0.0, 0.0)
    p = count / total
    z = 1.96
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denom
    return (max(0.0, center - spread), min(1.0, center + spread))
