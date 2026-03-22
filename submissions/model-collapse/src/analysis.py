"""Post-experiment analysis: aggregate metrics, curve fitting, summaries.

Computes:
  - Mean KL divergence per (agent_type, gt_fraction, generation) across seeds
  - Collapse rate: mean generation at which KL > threshold
  - Quality curve classification: exponential vs linear vs stable
  - Anchor effectiveness: marginal collapse-delay per unit of gt_fraction
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_by_condition(
    results: list[dict[str, Any]],
) -> dict[tuple[str, float, str], dict[str, Any]]:
    """Group results by (agent_type, gt_fraction, dist_name) and average.

    Returns a dict keyed by condition tuple, each value containing:
      - mean_kl: array of mean KL per generation
      - std_kl: array of std KL per generation
      - mean_wd: array of mean Wasserstein per generation
      - std_wd: array of std Wasserstein per generation
      - collapse_generations: list of collapse gen (None replaced with n_generations)
      - mean_collapse: mean collapse generation
    """
    groups: dict[tuple[str, float, str], list[dict]] = {}
    for r in results:
        c = r["config"]
        key = (c["agent_type"], c["gt_fraction"], c["dist_name"])
        groups.setdefault(key, []).append(r)

    aggregated = {}
    for key, runs in groups.items():
        n_gen = len(runs[0]["generations"])
        kl_matrix = np.array([
            [g["kl_divergence"] for g in r["generations"]] for r in runs
        ])
        wd_matrix = np.array([
            [g["wasserstein_distance"] for g in r["generations"]] for r in runs
        ])

        collapse_gens = [
            r["collapse_generation"] if r["collapse_generation"] is not None else n_gen
            for r in runs
        ]

        aggregated[key] = {
            "mean_kl": kl_matrix.mean(axis=0),
            "std_kl": kl_matrix.std(axis=0),
            "mean_wd": wd_matrix.mean(axis=0),
            "std_wd": wd_matrix.std(axis=0),
            "collapse_generations": collapse_gens,
            "mean_collapse": float(np.mean(collapse_gens)),
            "n_generations": n_gen,
        }
    return aggregated


# ---------------------------------------------------------------------------
# Curve classification
# ---------------------------------------------------------------------------

def _exp_func(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


def _linear_func(x: np.ndarray, a: float, b: float) -> np.ndarray:
    return a * x + b


def classify_curve(mean_kl: np.ndarray) -> dict[str, Any]:
    """Classify a KL-divergence curve as exponential, linear, or stable.

    Returns dict with 'shape' in {'exponential', 'linear', 'stable'}
    and fit parameters.
    """
    x = np.arange(len(mean_kl), dtype=float)
    y = mean_kl.copy()

    # Stable: KL stays below 0.5 throughout
    if y.max() < 0.5:
        return {"shape": "stable", "max_kl": float(y.max())}

    # Try exponential fit
    exp_rmse = np.inf
    exp_params = None
    try:
        popt, _ = curve_fit(_exp_func, x, y, p0=[0.01, 0.3, y[0]], maxfev=5000)
        pred = _exp_func(x, *popt)
        exp_rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
        exp_params = {"a": float(popt[0]), "b": float(popt[1]), "c": float(popt[2])}
    except (RuntimeError, ValueError):
        pass

    # Try linear fit
    try:
        popt_l, _ = curve_fit(_linear_func, x, y, p0=[0.1, y[0]])
        pred_l = _linear_func(x, *popt_l)
        lin_rmse = float(np.sqrt(np.mean((y - pred_l) ** 2)))
        lin_params = {"a": float(popt_l[0]), "b": float(popt_l[1])}
    except (RuntimeError, ValueError):
        lin_rmse = np.inf
        lin_params = None

    if exp_rmse < lin_rmse and exp_params is not None and exp_params["b"] > 0.05:
        return {"shape": "exponential", "rmse": exp_rmse, "params": exp_params}
    elif lin_params is not None:
        return {"shape": "linear", "rmse": lin_rmse, "params": lin_params}
    else:
        return {"shape": "stable", "max_kl": float(y.max())}


# ---------------------------------------------------------------------------
# Anchor effectiveness
# ---------------------------------------------------------------------------

def anchor_effectiveness(
    aggregated: dict[tuple[str, float, str], dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """For each (agent_type, dist_name), measure how gt_fraction delays collapse.

    Returns {agent_type: [{dist_name, gt_fraction, mean_collapse, delta_per_pct}, ...]}.
    """
    by_agent: dict[str, list[dict[str, Any]]] = {}
    for (at, gf, dn), agg in aggregated.items():
        by_agent.setdefault(at, []).append({
            "dist_name": dn,
            "gt_fraction": gf,
            "mean_collapse": agg["mean_collapse"],
        })

    # Sort by gt_fraction within each agent and compute marginal effectiveness
    for at, entries in by_agent.items():
        entries.sort(key=lambda e: (e["dist_name"], e["gt_fraction"]))
        for i, e in enumerate(entries):
            if e["gt_fraction"] == 0.0:
                e["delta_per_pct"] = None
            else:
                # Find baseline (same agent, same dist, gt_fraction=0)
                baseline = [
                    x for x in entries
                    if x["dist_name"] == e["dist_name"] and x["gt_fraction"] == 0.0
                ]
                if baseline:
                    delay = e["mean_collapse"] - baseline[0]["mean_collapse"]
                    e["delta_per_pct"] = delay / (e["gt_fraction"] * 100)
                else:
                    e["delta_per_pct"] = None

    return by_agent


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def build_summary(
    aggregated: dict[tuple[str, float, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build a flat summary table for reporting."""
    rows = []
    for (at, gf, dn), agg in sorted(aggregated.items()):
        curve = classify_curve(agg["mean_kl"])
        rows.append({
            "agent_type": at,
            "gt_fraction": gf,
            "dist_name": dn,
            "final_kl_mean": float(agg["mean_kl"][-1]),
            "final_kl_std": float(agg["std_kl"][-1]),
            "final_wd_mean": float(agg["mean_wd"][-1]),
            "mean_collapse_gen": agg["mean_collapse"],
            "curve_shape": curve["shape"],
        })
    return rows
