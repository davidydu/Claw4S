"""Full experiment runner with multiprocessing.

Runs the 4 algorithms x 3 strategies x 5 Sybil counts x 3 seeds grid.
K=0 baselines use strategy="none" (no attack), giving
4 x 3 x 1 + 4 x 3 x 4 x 3 = 12 + 144 = 156 simulations.
"""

from __future__ import annotations

import json
import os
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List

from .simulation import run_single_sim
from .reputation import ALGORITHMS
from .sybil_strategies import STRATEGIES

N_HONEST = 20
SYBIL_COUNTS = [0, 2, 5, 10, 20]
SEEDS = [42, 123, 7]
N_ROUNDS = 5000


def _build_task_list() -> List[Dict]:
    """Build the full grid of simulation tasks."""
    tasks = []
    for algo in sorted(ALGORITHMS.keys()):
        for k in SYBIL_COUNTS:
            if k == 0:
                # Baseline: no attack, strategy is irrelevant
                for seed in SEEDS:
                    tasks.append({
                        "n_honest": N_HONEST,
                        "n_sybil": 0,
                        "algorithm_name": algo,
                        "strategy_name": "none",
                        "n_rounds": N_ROUNDS,
                        "seed": seed,
                    })
            else:
                for strategy in sorted(STRATEGIES.keys()):
                    for seed in SEEDS:
                        tasks.append({
                            "n_honest": N_HONEST,
                            "n_sybil": k,
                            "algorithm_name": algo,
                            "strategy_name": strategy,
                            "n_rounds": N_ROUNDS,
                            "seed": seed,
                        })
    return tasks


def _run_task(task: Dict) -> Dict:
    """Worker function for multiprocessing."""
    return run_single_sim(**task)


def run_experiment(n_workers: int = 0) -> Dict:
    """Run the full experiment grid.

    Args:
        n_workers: Number of parallel workers. 0 = auto (cpu_count).

    Returns:
        Dict with metadata and list of all simulation results.
    """
    if n_workers <= 0:
        n_workers = max(1, cpu_count())

    tasks = _build_task_list()
    total = len(tasks)

    print(f"[1/3] Running {total} simulations with {n_workers} workers...")
    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_task, tasks)

    elapsed = time.time() - t0
    print(f"[2/3] Completed in {elapsed:.1f}s")

    # Aggregate
    output = {
        "metadata": {
            "n_honest": N_HONEST,
            "sybil_counts": SYBIL_COUNTS,
            "algorithms": sorted(ALGORITHMS.keys()),
            "strategies": sorted(STRATEGIES.keys()),
            "seeds": SEEDS,
            "n_rounds": N_ROUNDS,
            "total_simulations": total,
            "elapsed_seconds": round(elapsed, 2),
            "n_workers": n_workers,
        },
        "results": results,
    }

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"[3/3] Saved results to results/results.json")

    return output


def run_diagnostic(n_workers: int = 0) -> Dict:
    """Run a small diagnostic (2 algos x 1 strategy x 2 K values x 1 seed).

    Used to sanity-check before the full experiment.
    """
    if n_workers <= 0:
        n_workers = max(1, cpu_count())

    tasks = []
    for algo in ["simple_average", "eigentrust"]:
        for k in [0, 5]:
            strategy = "none" if k == 0 else "ballot_stuffing"
            tasks.append({
                "n_honest": N_HONEST,
                "n_sybil": k,
                "algorithm_name": algo,
                "strategy_name": strategy,
                "n_rounds": 1000,
                "seed": 42,
            })

    print(f"[diagnostic] Running {len(tasks)} tasks with {n_workers} workers...")
    t0 = time.time()

    with Pool(processes=n_workers) as pool:
        results = pool.map(_run_task, tasks)

    elapsed = time.time() - t0
    print(f"[diagnostic] Done in {elapsed:.1f}s")

    for r in results:
        cfg = r["config"]
        m = r["metrics"]
        print(
            f"  {cfg['algorithm']:20s} K={cfg['n_sybil']:2d} "
            f"accuracy={m['reputation_accuracy']:.3f} "
            f"detection={m['sybil_detection_rate']:.3f} "
            f"welfare={m['honest_welfare']:.3f} "
            f"efficiency={m['market_efficiency']:.3f}"
        )

    return {"diagnostic_results": results, "elapsed": elapsed}
