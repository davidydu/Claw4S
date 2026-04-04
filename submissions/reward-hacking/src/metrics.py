"""Metrics computation for reward-hacking propagation experiments.

Metrics
-------
1. Hack adoption rate: fraction of agents using the hack (from time-series)
2. Propagation speed: rounds until 50% / 90% adoption
3. Containment success: binary — did monitors prevent full spread?
4. Welfare impact: mean proxy-true reward divergence at steady state
5. Time-to-detection: rounds until a monitor flags divergence (first
   quarantine event or detection threshold breach)
"""

from __future__ import annotations

import numpy as np


def compute_summary_metrics(sim_result: dict) -> dict:
    """Compute summary metrics from a single simulation result.

    Parameters
    ----------
    sim_result : dict
        Output of run_simulation().

    Returns
    -------
    metrics : dict
    """
    adoption = sim_result["adoption_curve"]
    n_rounds = len(adoption)

    # Steady-state adoption: mean over last 20% of rounds
    tail_start = int(n_rounds * 0.8)
    steady_adoption = float(np.mean(adoption[tail_start:])) if n_rounds > 0 else 0.0

    # Steady-state welfare impact: mean divergence over last 20%
    divergence = sim_result["divergence_curve"]
    steady_divergence = float(np.mean(divergence[tail_start:])) if n_rounds > 0 else 0.0

    # Containment: hack never reached 100% of non-monitor agents
    contained = sim_result["final_adoption"] < 1.0

    # Peak adoption rate
    peak_adoption = float(np.max(adoption)) if adoption else 0.0

    return {
        "steady_state_adoption": round(steady_adoption, 4),
        "peak_adoption": round(peak_adoption, 4),
        "time_to_50pct": sim_result["time_to_50pct"],
        "time_to_90pct": sim_result["time_to_90pct"],
        "containment_success": contained,
        "containment_events": sim_result["containment_events"],
        "steady_state_divergence": round(steady_divergence, 4),
        "final_adoption": round(sim_result["final_adoption"], 4),
    }


def aggregate_across_seeds(
    results: list[dict],
) -> dict:
    """Aggregate summary metrics across multiple seeds.

    Parameters
    ----------
    results : list of dict
        Each dict is the output of compute_summary_metrics().

    Returns
    -------
    agg : dict with mean and std for each numeric metric.
    """
    if not results:
        return {}

    keys = [
        "steady_state_adoption",
        "peak_adoption",
        "steady_state_divergence",
        "final_adoption",
    ]

    agg: dict = {}
    for k in keys:
        vals = [r[k] for r in results]
        agg[f"{k}_mean"] = round(float(np.mean(vals)), 4)
        agg[f"{k}_std"] = round(float(np.std(vals)), 4)

    # Containment success rate
    contained = [1.0 if r["containment_success"] else 0.0 for r in results]
    agg["containment_rate"] = round(float(np.mean(contained)), 4)

    # Propagation speed: mean time_to_50pct (excluding None)
    t50 = [r["time_to_50pct"] for r in results if r["time_to_50pct"] is not None]
    agg["time_to_50pct_mean"] = round(float(np.mean(t50)), 1) if t50 else None
    agg["time_to_50pct_reached_frac"] = round(len(t50) / len(results), 4)

    t90 = [r["time_to_90pct"] for r in results if r["time_to_90pct"] is not None]
    agg["time_to_90pct_mean"] = round(float(np.mean(t90)), 1) if t90 else None
    agg["time_to_90pct_reached_frac"] = round(len(t90) / len(results), 4)

    # Total containment events
    ce = [r["containment_events"] for r in results]
    agg["containment_events_mean"] = round(float(np.mean(ce)), 1)

    return agg
