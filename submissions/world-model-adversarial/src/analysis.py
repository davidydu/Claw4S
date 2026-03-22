"""Aggregate and analyze results from multiple simulation runs.

Takes a list of SimResults and produces summary tables and statistics
ready for the research note.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.experiment import SimResult


def _group_key(result: SimResult) -> str:
    """Group key: learner-vs-adversary_regime_noise (no seed)."""
    c = result.config
    return f"{c.learner_code}-vs-{c.adversary_code}_{c.drift_regime}_noise{c.noise_level}"


def aggregate_results(results: list[SimResult]) -> dict[str, dict[str, Any]]:
    """Aggregate results across seeds for each unique configuration.

    Returns a dict keyed by group label, with mean and std for each
    audit metric across seeds.
    """
    groups: dict[str, list[SimResult]] = defaultdict(list)
    for r in results:
        groups[_group_key(r)].append(r)

    aggregated: dict[str, dict[str, Any]] = {}
    for key, group in sorted(groups.items()):
        agg: dict[str, Any] = {"n_seeds": len(group)}
        # Flatten all audit metrics.
        all_metrics: dict[str, list[float]] = defaultdict(list)
        for r in group:
            for section, metrics in r.audit.items():
                for metric_name, value in metrics.items():
                    full_key = f"{section}.{metric_name}"
                    all_metrics[full_key].append(value)

        for metric_key, values in sorted(all_metrics.items()):
            arr = np.array(values)
            agg[f"{metric_key}.mean"] = float(np.mean(arr))
            agg[f"{metric_key}.std"] = float(np.std(arr))
        aggregated[key] = agg

    return aggregated


def _parse_group_key(key: str) -> tuple[str, str, str, float]:
    """Parse group key into (learner, adversary, regime, noise).

    Key format: ``NL-vs-SA_stable_noise0.0`` or
    ``NL-vs-SA_slow_drift_noise0.1``.
    """
    import re
    m = re.match(
        r"^([A-Z]+)-vs-([A-Z]+)_(.+)_noise([\d.]+)$", key
    )
    if not m:
        raise ValueError(f"Cannot parse key: {key}")
    return m.group(1), m.group(2), m.group(3), float(m.group(4))


def build_summary_table(
    aggregated: dict[str, dict[str, Any]],
    metric: str = "distortion.mean_belief_error",
) -> list[dict[str, Any]]:
    """Build a summary table for one metric across all configurations.

    Returns a list of dicts, each with keys:
    learner, adversary, regime, noise, mean, std.
    """
    rows: list[dict[str, Any]] = []
    for key, agg in sorted(aggregated.items()):
        learner, adversary, regime, noise = _parse_group_key(key)
        rows.append({
            "learner": learner,
            "adversary": adversary,
            "regime": regime,
            "noise": noise,
            "mean": agg.get(f"{metric}.mean", float("nan")),
            "std": agg.get(f"{metric}.std", float("nan")),
        })
    return rows


def compute_manipulation_speed(
    results: list[SimResult],
    threshold: float = 0.8,
) -> dict[str, dict[str, float]]:
    """Compute rounds until belief error exceeds threshold.

    For each config group, find the first sampled round where
    belief error > threshold, averaged across seeds.
    """
    groups: dict[str, list[SimResult]] = defaultdict(list)
    for r in results:
        groups[_group_key(r)].append(r)

    speeds: dict[str, dict[str, float]] = {}
    for key, group in sorted(groups.items()):
        first_exceeds: list[float] = []
        for r in group:
            interval = r.config.belief_sample_interval
            found = False
            for i, err in enumerate(r.belief_error_timeseries):
                if err > threshold:
                    first_exceeds.append(float(i * interval))
                    found = True
                    break
            if not found:
                first_exceeds.append(float(r.config.n_rounds))
        speeds[key] = {
            "mean_rounds": float(np.mean(first_exceeds)),
            "std_rounds": float(np.std(first_exceeds)),
        }
    return speeds


def compute_resilience_ranking(
    aggregated: dict[str, dict[str, Any]],
    adversary_code: str = "SA",
    noise: float = 0.0,
) -> list[dict[str, Any]]:
    """Rank learners by resilience against a specific adversary.

    Lower final belief error = more resilient.
    """
    rows: list[dict[str, Any]] = []
    for key, agg in aggregated.items():
        if f"-vs-{adversary_code}_" not in key:
            continue
        if f"noise{noise}" not in key:
            continue
        parts = key.split("_")
        matchup = parts[0]
        regime = parts[1]
        learner = matchup.split("-vs-")[0]
        rows.append({
            "learner": learner,
            "regime": regime,
            "final_belief_error_mean": agg.get(
                "distortion.final_belief_error.mean", float("nan")
            ),
            "accuracy_mean": agg.get(
                "decision_quality.accuracy.mean", float("nan")
            ),
        })
    return sorted(rows, key=lambda r: r["final_belief_error_mean"])
