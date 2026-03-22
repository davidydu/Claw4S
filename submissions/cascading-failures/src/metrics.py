"""Aggregate metrics and statistical analysis of experiment results.

Computes per-condition means and standard deviations across seeds,
and performs key comparisons (hub vs random attack, topology ranking, etc.).
"""

from __future__ import annotations

import math
from typing import Dict, List, Any, Tuple


def _mean(values: List[float]) -> float:
    """Mean of a list, treating inf as NaN (excluded)."""
    finite = [v for v in values if math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def _std(values: List[float]) -> float:
    """Sample standard deviation, excluding inf."""
    finite = [v for v in values if math.isfinite(v)]
    if len(finite) < 2:
        return 0.0
    m = sum(finite) / len(finite)
    var = sum((v - m) ** 2 for v in finite) / (len(finite) - 1)
    return math.sqrt(var)


def aggregate_by_condition(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Group results by (topology, agent_type, shock_magnitude, shock_location)
    and compute mean +/- std across seeds.

    Returns list of dicts with aggregated metrics.
    """
    groups: Dict[Tuple, List[Dict]] = {}
    for r in results:
        key = (r["topology"], r["agent_type"], r["shock_magnitude"], r["shock_location"])
        groups.setdefault(key, []).append(r)

    aggregated = []
    for (topo, atype, mag, loc), group in sorted(groups.items()):
        agg = {
            "topology": topo,
            "agent_type": atype,
            "shock_magnitude": mag,
            "shock_location": loc,
            "n_seeds": len(group),
            "cascade_size_mean": _mean([r["cascade_size"] for r in group]),
            "cascade_size_std": _std([r["cascade_size"] for r in group]),
            "cascade_speed_mean": _mean([r["cascade_speed"] for r in group]),
            "cascade_speed_std": _std([r["cascade_speed"] for r in group]),
            "recovery_time_mean": _mean([r["recovery_time"] for r in group]),
            "recovery_time_std": _std([r["recovery_time"] for r in group]),
            "systemic_risk_mean": _mean([r["systemic_risk"] for r in group]),
            "systemic_risk_std": _std([r["systemic_risk"] for r in group]),
        }
        aggregated.append(agg)
    return aggregated


def topology_ranking(aggregated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Rank topologies by mean systemic risk (averaged over all conditions).

    Returns list of dicts sorted by risk (highest first).
    """
    topo_risks: Dict[str, List[float]] = {}
    for a in aggregated:
        risk = a["systemic_risk_mean"]
        if math.isfinite(risk):
            topo_risks.setdefault(a["topology"], []).append(risk)

    ranking = []
    for topo, risks in topo_risks.items():
        ranking.append({
            "topology": topo,
            "mean_systemic_risk": _mean(risks),
            "std_systemic_risk": _std(risks),
            "n_conditions": len(risks),
        })
    ranking.sort(key=lambda x: x["mean_systemic_risk"], reverse=True)
    return ranking


def hub_vs_random_comparison(aggregated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare hub-attack vs random-attack cascade sizes per topology.

    Returns list of dicts with hub_cascade, random_cascade, and ratio.
    """
    comparisons: Dict[str, Dict[str, List[float]]] = {}
    for a in aggregated:
        topo = a["topology"]
        loc = a["shock_location"]
        cs = a["cascade_size_mean"]
        if math.isfinite(cs):
            comparisons.setdefault(topo, {}).setdefault(loc, []).append(cs)

    results = []
    for topo in sorted(comparisons.keys()):
        hub_vals = comparisons[topo].get("hub", [])
        rand_vals = comparisons[topo].get("random", [])
        hub_mean = _mean(hub_vals) if hub_vals else float("nan")
        rand_mean = _mean(rand_vals) if rand_vals else float("nan")
        ratio = hub_mean / rand_mean if rand_mean and math.isfinite(rand_mean) and rand_mean > 0 else float("nan")
        results.append({
            "topology": topo,
            "hub_cascade_size": hub_mean,
            "random_cascade_size": rand_mean,
            "hub_to_random_ratio": ratio,
        })
    return results


def agent_type_comparison(aggregated: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compare agent types by mean cascade size (averaged over all conditions).

    Returns list sorted by cascade size (lowest = most resilient first).
    """
    agent_cascades: Dict[str, List[float]] = {}
    for a in aggregated:
        cs = a["cascade_size_mean"]
        if math.isfinite(cs):
            agent_cascades.setdefault(a["agent_type"], []).append(cs)

    results = []
    for atype, cascades in agent_cascades.items():
        results.append({
            "agent_type": atype,
            "mean_cascade_size": _mean(cascades),
            "std_cascade_size": _std(cascades),
            "n_conditions": len(cascades),
        })
    results.sort(key=lambda x: x["mean_cascade_size"])
    return results
