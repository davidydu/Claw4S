"""
Statistical analysis utilities and phase-transition detection.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.auditors import AuditResult


@dataclass
class AggregatedMetric:
    """Mean +/- std over seeds for a single metric."""
    mean: float
    std: float
    values: List[float]


def aggregate_over_seeds(
    audit_results: List[Dict[str, AuditResult]],
) -> Dict[str, AggregatedMetric]:
    """Given audit dicts from multiple seeds, compute mean/std per metric."""
    metrics: Dict[str, List[float]] = {}
    for ar in audit_results:
        for name, res in ar.items():
            metrics.setdefault(name, []).append(res.value)

    out = {}
    for name, vals in metrics.items():
        arr = np.array(vals)
        out[name] = AggregatedMetric(
            mean=float(arr.mean()),
            std=float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            values=vals,
        )
    return out


# ------------------------------------------------------------------
# Phase-transition detection
# ------------------------------------------------------------------

def detect_phase_transition(
    disagreement_levels: List[float],
    coordination_rates: List[float],
    threshold: float = 0.5,
) -> Optional[float]:
    """Find the disagreement level where coordination drops below *threshold*.

    Uses linear interpolation between the two bracketing points.
    Returns None if coordination never drops below threshold, or if it
    starts below threshold.
    """
    d = np.array(disagreement_levels)
    c = np.array(coordination_rates)

    # Sort by disagreement
    order = np.argsort(d)
    d, c = d[order], c[order]

    # If first point already below threshold, transition is at or before d[0]
    if c[0] < threshold:
        return float(d[0])

    # Scan for crossing
    for i in range(len(d) - 1):
        if c[i] >= threshold and c[i + 1] < threshold:
            # Linear interpolation
            frac = (threshold - c[i]) / (c[i + 1] - c[i])
            return float(d[i] + frac * (d[i + 1] - d[i]))

    return None  # never drops below threshold


def compute_sharpness(
    disagreement_levels: List[float],
    coordination_rates: List[float],
) -> float:
    """Quantify how sharp the phase transition is.

    Returns the maximum absolute discrete derivative of coordination_rate
    with respect to disagreement.  Larger = sharper transition.
    """
    d = np.array(disagreement_levels)
    c = np.array(coordination_rates)

    order = np.argsort(d)
    d, c = d[order], c[order]

    if len(d) < 2:
        return 0.0

    derivatives = np.abs(np.diff(c) / np.diff(d))
    return float(derivatives.max())


def build_summary_table(
    results_by_key: Dict[Tuple[str, int, float], Dict[str, AggregatedMetric]],
) -> List[Dict]:
    """Build a flat table of results for reporting.

    Key = (composition, n_agents, disagreement).
    """
    rows = []
    for (comp, n, d), metrics in sorted(results_by_key.items()):
        row = {
            "composition": comp,
            "n_agents": n,
            "disagreement": d,
        }
        for mname, agg in metrics.items():
            row[f"{mname}_mean"] = round(agg.mean, 4)
            row[f"{mname}_std"] = round(agg.std, 4)
        rows.append(row)
    return rows
