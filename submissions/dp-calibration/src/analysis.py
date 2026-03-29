"""Run the full DP noise calibration comparison across parameter grid.

Sweeps over all combinations of (T, delta, sigma) and computes epsilon
using all four privacy accounting methods. Produces structured results
for visualization and validation.
"""

import hashlib
import json
import os
import platform
import sys
import time
from importlib import metadata as importlib_metadata

import numpy as np

from src.accounting import METHOD_NAMES, compute_all_epsilons


# Parameter grid
T_VALUES = [10, 100, 1000, 10000]
DELTA_VALUES = [1e-5, 1e-6, 1e-7]
SIGMA_VALUES = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]


def _normalize_t_values(values: list[int]) -> list[int]:
    """Validate/normalize T grid values."""
    if not values:
        raise ValueError("t_values must be non-empty")
    normalized = []
    for value in values:
        T = int(value)
        if T <= 0:
            raise ValueError(f"Invalid T={value}; must be > 0")
        normalized.append(T)
    return normalized


def _normalize_delta_values(values: list[float]) -> list[float]:
    """Validate/normalize delta grid values."""
    if not values:
        raise ValueError("delta_values must be non-empty")
    normalized = []
    for value in values:
        delta = float(value)
        if not (0.0 < delta < 1.0):
            raise ValueError(f"Invalid delta={value}; must satisfy 0 < delta < 1")
        normalized.append(delta)
    return normalized


def _normalize_sigma_values(values: list[float]) -> list[float]:
    """Validate/normalize sigma grid values."""
    if not values:
        raise ValueError("sigma_values must be non-empty")
    normalized = []
    for value in values:
        sigma = float(value)
        if sigma <= 0.0:
            raise ValueError(f"Invalid sigma={value}; must be > 0")
        normalized.append(sigma)
    return normalized


def _canonical_float(value: float | int) -> float | str:
    """Normalize numbers so digest is stable across platforms."""
    number = float(value)
    if number == float("inf"):
        return "Infinity"
    if number == float("-inf"):
        return "-Infinity"
    if number != number:
        return "NaN"
    return round(number, 12)


def compute_results_digest(results: list[dict]) -> str:
    """Compute a deterministic SHA256 digest of analysis results."""
    canonical_results = []
    sorted_results = sorted(results, key=lambda r: (r["T"], r["delta"], r["sigma"]))

    for row in sorted_results:
        eps = {
            method: _canonical_float(value)
            for method, value in sorted(row["epsilons"].items())
        }
        ratios = {
            method: _canonical_float(value)
            for method, value in sorted(row["tightness_ratio"].items())
        }
        canonical_results.append({
            "T": int(row["T"]),
            "delta": _canonical_float(row["delta"]),
            "sigma": _canonical_float(row["sigma"]),
            "best_method": row["best_method"],
            "best_epsilon": _canonical_float(row["best_epsilon"]),
            "epsilons": eps,
            "tightness_ratio": ratios,
        })

    payload = json.dumps(canonical_results, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _collect_package_versions() -> dict[str, str]:
    """Capture runtime package versions for reproducibility manifests."""
    packages = ("numpy", "scipy", "matplotlib")
    return {pkg: importlib_metadata.version(pkg) for pkg in packages}


def run_analysis(
    seed: int = 42,
    t_values: list[int] | None = None,
    delta_values: list[float] | None = None,
    sigma_values: list[float] | None = None,
) -> dict:
    """Run the full parameter sweep and return structured results.

    Returns a dict with:
        - metadata: grid sizes, method names, timing info
        - grid: the parameter values used
        - results: list of dicts, one per (T, delta, sigma) combination
        - summary: tightness analysis and method rankings
    """
    np.random.seed(seed)
    t_grid = _normalize_t_values(list(T_VALUES if t_values is None else t_values))
    delta_grid = _normalize_delta_values(
        list(DELTA_VALUES if delta_values is None else delta_values)
    )
    sigma_grid = _normalize_sigma_values(
        list(SIGMA_VALUES if sigma_values is None else sigma_values)
    )

    start_time = time.time()

    results = []
    for T in t_grid:
        for delta in delta_grid:
            for sigma in sigma_grid:
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
    summary = _compute_summary(results, t_grid)

    return {
        "metadata": {
            "num_T": len(t_grid),
            "num_delta": len(delta_grid),
            "num_sigma": len(sigma_grid),
            "num_methods": len(METHOD_NAMES),
            "total_configs": len(results),
            "total_computations": len(results) * len(METHOD_NAMES),
            "elapsed_seconds": round(elapsed, 3),
            "seed": seed,
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "package_versions": _collect_package_versions(),
            "results_digest": compute_results_digest(results),
        },
        "grid": {
            "T_values": t_grid,
            "delta_values": delta_grid,
            "sigma_values": sigma_grid,
            "methods": METHOD_NAMES,
        },
        "results": results,
        "summary": summary,
    }


def _compute_summary(results: list[dict], t_values: list[int]) -> dict:
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
    median_tightness = {}
    p95_tightness = {}
    for m in METHOD_NAMES:
        vals = tightness_lists[m]
        if vals:
            avg_tightness[m] = round(float(np.mean(vals)), 4)
            std_tightness[m] = round(float(np.std(vals)), 4)
            median_tightness[m] = round(float(np.median(vals)), 4)
            p95_tightness[m] = round(float(np.quantile(vals, 0.95)), 4)
        else:
            avg_tightness[m] = float("inf")
            std_tightness[m] = 0.0
            median_tightness[m] = float("inf")
            p95_tightness[m] = float("inf")

    # Method wins by regime (T value)
    wins_by_T = {}
    for T in t_values:
        wins_by_T[str(T)] = {m: 0 for m in METHOD_NAMES}
        for r in results:
            if r["T"] == T and r["best_method"] != "none":
                wins_by_T[str(T)][r["best_method"]] += 1

    return {
        "win_counts": win_counts,
        "avg_tightness_ratio": avg_tightness,
        "std_tightness_ratio": std_tightness,
        "median_tightness_ratio": median_tightness,
        "p95_tightness_ratio": p95_tightness,
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

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=_serialize)

    return path
