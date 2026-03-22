"""Experiment runner: sweeps over compositions, competition, complementarity, seeds.

Uses multiprocessing for parallel execution.
"""

from __future__ import annotations

import json
import os
import multiprocessing as mp
from functools import partial
from pathlib import Path

from src.simulation import run_simulation

# ---- Experimental Design ----

COMPOSITIONS = {
    "all_open": ["open", "open", "open", "open"],
    "all_secretive": ["secretive", "secretive", "secretive", "secretive"],
    "mixed": ["open", "secretive", "reciprocal", "strategic"],
    "all_strategic": ["strategic", "strategic", "strategic", "strategic"],
}

COMPETITION_LEVELS = {"low": 0.2, "medium": 0.5, "high": 0.8}

COMPLEMENTARITY_LEVELS = {"low": 0.3, "medium": 0.6, "high": 0.9}

SEEDS = [42, 123, 7]

N_ROUNDS = 10_000


def _build_tasks() -> list[dict]:
    """Build all simulation task configs."""
    tasks = []
    for comp_name, comp in COMPOSITIONS.items():
        for cl_name, cl_val in COMPETITION_LEVELS.items():
            for cm_name, cm_val in COMPLEMENTARITY_LEVELS.items():
                for seed in SEEDS:
                    tasks.append({
                        "composition_name": comp_name,
                        "composition": comp,
                        "competition_name": cl_name,
                        "competition": cl_val,
                        "complementarity_name": cm_name,
                        "complementarity": cm_val,
                        "seed": seed,
                    })
    return tasks


def _run_task(task: dict) -> dict:
    """Execute a single simulation task (for multiprocessing)."""
    result = run_simulation(
        composition=task["composition"],
        competition=task["competition"],
        complementarity=task["complementarity"],
        n_rounds=N_ROUNDS,
        seed=task["seed"],
    )
    result["labels"] = {
        "composition": task["composition_name"],
        "competition": task["competition_name"],
        "complementarity": task["complementarity_name"],
    }
    return result


def run_experiment(
    n_workers: int | None = None,
    results_dir: str = "results",
) -> list[dict]:
    """Run the full experiment grid with multiprocessing.

    Parameters
    ----------
    n_workers : int or None
        Number of parallel workers. Defaults to cpu_count.
    results_dir : str
        Directory to save results.

    Returns
    -------
    list[dict] — all simulation results.
    """
    tasks = _build_tasks()
    n_total = len(tasks)
    print(f"[1/3] Running {n_total} simulations ({N_ROUNDS} rounds each)...")

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_task, tasks)

    print(f"[2/3] All {n_total} simulations complete.")

    # Save results
    os.makedirs(results_dir, exist_ok=True)
    out_path = os.path.join(results_dir, "results.json")
    output = {
        "metadata": {
            "n_simulations": n_total,
            "n_rounds": N_ROUNDS,
            "compositions": list(COMPOSITIONS.keys()),
            "competition_levels": COMPETITION_LEVELS,
            "complementarity_levels": COMPLEMENTARITY_LEVELS,
            "seeds": SEEDS,
            "n_agents": 4,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"[3/3] Results saved to {out_path}")

    return results
