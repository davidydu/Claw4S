"""Statistical analysis of experiment results.

Aggregates across seeds, computes means/stds, identifies phase transitions,
and ranks agent types.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


def load_results(path: str = "results/results.json") -> dict:
    """Load experiment results from JSON."""
    with open(path) as f:
        return json.load(f)


def aggregate_by_condition(results: list[dict]) -> dict:
    """Aggregate simulation results by experimental condition (comp x compet x compl).

    Returns dict keyed by (composition, competition, complementarity) with
    mean and std across seeds for each summary metric.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in results:
        key = (
            r["labels"]["composition"],
            r["labels"]["competition"],
            r["labels"]["complementarity"],
        )
        groups[key].append(r["summary"])

    aggregated = {}
    for key, summaries in groups.items():
        agg = {}
        for metric in [
            "avg_sharing_rate",
            "avg_group_welfare",
            "avg_welfare_gap",
            "avg_info_asymmetry",
            "tail_sharing_rate",
            "tail_group_welfare",
            "norm_convergence_std",
        ]:
            vals = [s[metric] for s in summaries]
            agg[metric] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

        # Per-type sharing
        type_sharing: dict[str, list[float]] = defaultdict(list)
        for s in summaries:
            for t, v in s["per_type_sharing"].items():
                type_sharing[t].append(v)
        agg["per_type_sharing"] = {
            t: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for t, v in type_sharing.items()
        }

        # Per-type cumulative payoff
        type_payoffs: dict[str, list[float]] = defaultdict(list)
        for s in summaries:
            for t, v in s["per_type_cumulative_payoff"].items():
                type_payoffs[t].append(v)
        agg["per_type_payoff"] = {
            t: {"mean": float(np.mean(v)), "std": float(np.std(v))}
            for t, v in type_payoffs.items()
        }

        aggregated[key] = agg

    return aggregated


def find_phase_transition(aggregated: dict) -> dict:
    """Identify the competition level where sharing drops below 0.5 for mixed composition.

    Returns dict with phase transition info.
    """
    transitions = {}
    for comp_name in ["mixed", "all_strategic"]:
        for compl_name in ["low", "medium", "high"]:
            sharing_by_compet = {}
            for (c, co, cm), agg in aggregated.items():
                if c == comp_name and cm == compl_name:
                    sharing_by_compet[co] = agg["tail_sharing_rate"]["mean"]

            # Check if there's a transition across competition levels
            levels = ["low", "medium", "high"]
            vals = [sharing_by_compet.get(l, None) for l in levels]
            if all(v is not None for v in vals):
                transitions[f"{comp_name}_{compl_name}"] = {
                    "competition_levels": levels,
                    "tail_sharing_rates": vals,
                    "transition_detected": any(
                        vals[i] >= 0.5 and vals[i + 1] < 0.5
                        for i in range(len(vals) - 1)
                    ),
                }

    return transitions


def rank_agent_types(aggregated: dict) -> dict:
    """Rank agent types by average payoff and sharing behavior across conditions."""
    type_payoffs: dict[str, list[float]] = defaultdict(list)
    type_sharing: dict[str, list[float]] = defaultdict(list)

    for key, agg in aggregated.items():
        for t, v in agg["per_type_payoff"].items():
            type_payoffs[t].append(v["mean"])
        for t, v in agg["per_type_sharing"].items():
            type_sharing[t].append(v["mean"])

    rankings = {}
    for t in type_payoffs:
        rankings[t] = {
            "avg_payoff": float(np.mean(type_payoffs[t])),
            "avg_sharing": float(np.mean(type_sharing.get(t, [0.0]))),
        }

    # Sort by payoff descending
    sorted_types = sorted(rankings.items(), key=lambda x: x[1]["avg_payoff"], reverse=True)
    return {t: {**v, "rank": i + 1} for i, (t, v) in enumerate(sorted_types)}


def run_analysis(results_path: str = "results/results.json") -> dict:
    """Run full statistical analysis on experiment results.

    Returns dict with aggregated results, phase transitions, and rankings.
    """
    data = load_results(results_path)
    results = data["results"]
    metadata = data["metadata"]

    aggregated = aggregate_by_condition(results)
    transitions = find_phase_transition(aggregated)
    rankings = rank_agent_types(aggregated)

    # Serialize aggregated keys to strings for JSON
    aggregated_serial = {
        f"{k[0]}|{k[1]}|{k[2]}": v for k, v in aggregated.items()
    }

    return {
        "metadata": metadata,
        "aggregated": aggregated_serial,
        "phase_transitions": transitions,
        "agent_rankings": rankings,
    }
