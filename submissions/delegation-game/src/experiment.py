"""Experiment runner: sweep over incentive schemes, worker compositions,
noise levels, and seeds. Uses multiprocessing for parallelism.

Grid: 4 schemes x 4 worker compositions x 3 noise levels x 3 seeds = 144 sims
Each sim: 10,000 rounds with 3 workers.
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from src.simulation import SimConfig, SimResult, run_single_sim


# --- Experimental grid ---

INCENTIVE_SCHEMES = ["fixed_pay", "piece_rate", "tournament", "reputation"]

# 4 worker team compositions (3 workers each)
WORKER_COMPOSITIONS = {
    "all_honest": ["honest", "honest", "honest"],
    "all_strategic": ["strategic", "strategic", "strategic"],
    "mixed_honest_shirker": ["honest", "shirker", "strategic"],
    "all_adaptive": ["adaptive", "adaptive", "adaptive"],
}

NOISE_LEVELS = {
    "low": 0.5,
    "medium": 1.5,
    "high": 3.0,
}

SEEDS = [42, 123, 7]

NUM_ROUNDS = 10_000


def build_configs() -> list[SimConfig]:
    """Generate the full grid of simulation configurations."""
    configs = []
    for scheme in INCENTIVE_SCHEMES:
        for comp_name, worker_types in WORKER_COMPOSITIONS.items():
            for noise_name, noise_std in NOISE_LEVELS.items():
                for seed in SEEDS:
                    configs.append(SimConfig(
                        scheme_name=scheme,
                        worker_types=worker_types,
                        noise_std=noise_std,
                        num_rounds=NUM_ROUNDS,
                        seed=seed,
                    ))
    return configs


def _run_one(config: SimConfig) -> dict:
    """Wrapper for multiprocessing: run sim and return dict."""
    result = run_single_sim(config)
    return result.to_dict()


def run_experiment(output_dir: str = "results",
                   n_workers: int | None = None) -> dict:
    """Run the full experiment grid using multiprocessing.

    Returns the full results dictionary (also saved to disk).
    """
    configs = build_configs()
    n_sims = len(configs)

    if n_workers is None:
        n_workers = min(cpu_count(), 8)

    print(f"[1/3] Running {n_sims} simulations "
          f"({NUM_ROUNDS} rounds each, {n_workers} parallel workers)...")

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(_run_one, configs)
    elapsed = time.time() - t0

    print(f"[2/3] All simulations complete in {elapsed:.1f}s")

    # Aggregate across seeds: group by (scheme, worker_types, noise)
    aggregated = _aggregate_results(raw_results)

    output = {
        "metadata": {
            "num_simulations": n_sims,
            "num_rounds_per_sim": NUM_ROUNDS,
            "schemes": INCENTIVE_SCHEMES,
            "worker_compositions": {
                k: v for k, v in WORKER_COMPOSITIONS.items()
            },
            "noise_levels": {k: v for k, v in NOISE_LEVELS.items()},
            "seeds": SEEDS,
            "elapsed_seconds": round(elapsed, 1),
        },
        "raw_results": raw_results,
        "aggregated": aggregated,
    }

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "results.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[3/3] Saving results to {output_dir}/")

    return output


def _aggregate_results(raw: list[dict]) -> list[dict]:
    """Average metrics across seeds for each (scheme, composition, noise)."""
    import numpy as np

    groups: dict[str, list[dict]] = {}
    for r in raw:
        key = f"{r['scheme']}__{'-'.join(sorted(r['worker_types']))}__noise{r['noise_std']}"
        groups.setdefault(key, []).append(r)

    aggregated = []
    metric_keys = [
        "avg_quality", "principal_net_payoff", "worker_surplus",
        "shirking_rate", "quality_variance", "incentive_efficiency",
    ]
    for key, runs in sorted(groups.items()):
        agg = {
            "scheme": runs[0]["scheme"],
            "worker_types": runs[0]["worker_types"],
            "noise_std": runs[0]["noise_std"],
            "n_seeds": len(runs),
        }
        for m in metric_keys:
            vals = [r[m] for r in runs]
            agg[f"{m}_mean"] = round(float(np.mean(vals)), 4)
            agg[f"{m}_std"] = round(float(np.std(vals)), 4)
        aggregated.append(agg)
    return aggregated
