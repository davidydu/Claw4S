"""Run the full DP noise calibration comparison across parameter grid.

Sweeps over all combinations of (T, delta, sigma) and computes epsilon
using all four privacy accounting methods. Produces structured results
for visualization and validation.
"""

import json
import os
import time
from pathlib import Path

import numpy as np

from src.accounting import METHOD_NAMES, compute_all_epsilons


# Parameter grid
T_VALUES = [10, 100, 1000, 10000]
DELTA_VALUES = [1e-5, 1e-6, 1e-7]
SIGMA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def run_analysis(seed: int = 42) -> dict:
    """Run the full parameter sweep and return structured results.

    Returns a dict with:
        - metadata: grid sizes, method names, timing info
        - grid: the parameter values used
        - results: list of dicts, one per (T, delta, sigma) combination
        - summary: tightness analysis and method rankings
    """
    np.random.seed(seed)
    start_time = time.time()

    results = []
    for T in T_VALUES:
        for delta in DELTA_VALUES:
            for sigma in SIGMA_VALUES:
                epsilons = compute_all_epsilons(sigma=sigma, T=T, delta=delta)

                # Filter out infinite values for tightness computation
                finite_eps = {k: v for k, v in epsilons.items()
                              if v < float("inf")}

                if finite_eps:
                    best_eps = min(finite_eps.values())
                    best_method = min(finite_eps, key=finite_eps.get)
                    tightness = {k: v / best_eps if best_eps > 0 else float("inf")
                                 for k, v in finite_eps.items()}
                else:
                    best_eps = float("inf")
                    best_method = "none"
                    tightness = {}

                results.append({
                    "T": T,
                    "delta": delta,
                    "sigma": sigma,
                    "epsilons": epsilons,
                    "best_epsilon": best_eps,
                    "best_method": best_method,
                    "tightness_ratio": tightness,
                })

    elapsed = time.time() - start_time

    # Compute summary statistics
    summary = _compute_summary(results)

    return {
        "metadata": {
            "num_T": len(T_VALUES),
            "num_delta": len(DELTA_VALUES),
            "num_sigma": len(SIGMA_VALUES),
            "num_methods": len(METHOD_NAMES),
            "total_configs": len(results),
            "total_computations": len(results) * len(METHOD_NAMES),
            "elapsed_seconds": round(elapsed, 3),
            "seed": seed,
        },
        "grid": {
            "T_values": T_VALUES,
            "delta_values": DELTA_VALUES,
            "sigma_values": SIGMA_VALUES,
            "methods": METHOD_NAMES,
        },
        "results": results,
        "summary": summary,
    }


def _compute_summary(results: list[dict]) -> dict:
    """Compute summary statistics across all grid points."""
    # Count how often each method gives the tightest bound
    win_counts = {m: 0 for m in METHOD_NAMES}
    for r in results:
        if r["best_method"] != "none":
            win_counts[r["best_method"]] += 1

    # Average tightness ratio per method (excluding inf)
    tightness_lists = {m: [] for m in METHOD_NAMES}
    for r in results:
        for m in METHOD_NAMES:
            if m in r["tightness_ratio"]:
                ratio = r["tightness_ratio"][m]
                if ratio < float("inf"):
                    tightness_lists[m].append(ratio)

    avg_tightness = {}
    std_tightness = {}
    for m in METHOD_NAMES:
        vals = tightness_lists[m]
        if vals:
            avg_tightness[m] = round(float(np.mean(vals)), 4)
            std_tightness[m] = round(float(np.std(vals)), 4)
        else:
            avg_tightness[m] = float("inf")
            std_tightness[m] = 0.0

    # Method wins by regime (T value)
    wins_by_T = {}
    for T in T_VALUES:
        wins_by_T[str(T)] = {m: 0 for m in METHOD_NAMES}
        for r in results:
            if r["T"] == T and r["best_method"] != "none":
                wins_by_T[str(T)][r["best_method"]] += 1

    return {
        "win_counts": win_counts,
        "avg_tightness_ratio": avg_tightness,
        "std_tightness_ratio": std_tightness,
        "wins_by_T": wins_by_T,
    }


def save_results(data: dict, output_dir: str = "results") -> str:
    """Save results to JSON, handling non-serializable values."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "results.json")

    def _serialize(obj):
        if isinstance(obj, float):
            if obj == float("inf"):
                return "Infinity"
            if obj == float("-inf"):
                return "-Infinity"
            if obj != obj:  # NaN
                return "NaN"
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_serialize)

    return path
