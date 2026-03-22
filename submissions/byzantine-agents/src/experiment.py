"""Full experiment runner with multiprocessing.

Generates the 405-configuration grid and runs all simulations in parallel.
Grid: 3 honest types x 3 Byzantine strategies x 5 fractions x 3 committee sizes x 3 seeds.
"""

from __future__ import annotations

import json
import os
import time
from itertools import product
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np

from src.agents import HONEST_TYPES, BYZANTINE_TYPES
from src.simulation import SimConfig, SimResult, run_simulation
from src.metrics import (
    byzantine_threshold,
    byzantine_amplification,
    resilience_score,
)

HONEST_NAMES = list(HONEST_TYPES.keys())        # majority, bayesian, cautious
BYZANTINE_NAMES = list(BYZANTINE_TYPES.keys())   # random, strategic, mimicking
FRACTIONS = [0.0, 0.10, 0.20, 1 / 3, 0.50]     # Byzantine fractions
COMMITTEE_SIZES = [5, 9, 15]
SEEDS = [42, 123, 7]
ROUNDS_PER_SIM = 1_000


def _build_configs() -> list[SimConfig]:
    """Build the full 405-element configuration grid."""
    configs: list[SimConfig] = []
    for honest, byz, frac, n, seed in product(
        HONEST_NAMES, BYZANTINE_NAMES, FRACTIONS, COMMITTEE_SIZES, SEEDS
    ):
        configs.append(SimConfig(
            committee_size=n,
            honest_type=honest,
            byzantine_type=byz,
            byzantine_fraction=frac,
            rounds=ROUNDS_PER_SIM,
            seed=seed,
        ))
    return configs


def _run_one(cfg: SimConfig) -> dict[str, Any]:
    """Worker function — runs one simulation and returns a serializable dict."""
    result = run_simulation(cfg)
    return {
        "committee_size": cfg.committee_size,
        "honest_type": cfg.honest_type,
        "byzantine_type": cfg.byzantine_type,
        "byzantine_fraction": round(cfg.byzantine_fraction, 4),
        "seed": cfg.seed,
        "rounds": cfg.rounds,
        "accuracy": round(result.accuracy, 6),
        "accuracy_std": round(result.accuracy_std, 6),
        "num_honest": result.num_honest,
        "num_byzantine": result.num_byzantine,
    }


def _aggregate_results(raw: list[dict]) -> dict[str, Any]:
    """Aggregate raw per-seed results into summary statistics and derived metrics."""
    # Group by (honest_type, byzantine_type, byzantine_fraction, committee_size)
    from collections import defaultdict
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in raw:
        key = (r["honest_type"], r["byzantine_type"], r["byzantine_fraction"], r["committee_size"])
        groups[key].append(r)

    summaries = []
    for key, items in sorted(groups.items()):
        honest, byz, frac, n = key
        accs = [it["accuracy"] for it in items]
        summaries.append({
            "honest_type": honest,
            "byzantine_type": byz,
            "byzantine_fraction": frac,
            "committee_size": n,
            "mean_accuracy": round(float(np.mean(accs)), 6),
            "std_accuracy": round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 6),
            "n_seeds": len(items),
        })

    # Compute derived metrics per (honest_type, byzantine_type, committee_size)
    derived: dict[tuple, dict] = {}
    combo_groups: dict[tuple, list[dict]] = defaultdict(list)
    for s in summaries:
        combo_key = (s["honest_type"], s["byzantine_type"], s["committee_size"])
        combo_groups[combo_key].append(s)

    for combo_key, items_sorted in combo_groups.items():
        items_sorted = sorted(items_sorted, key=lambda x: x["byzantine_fraction"])
        fracs = [it["byzantine_fraction"] for it in items_sorted]
        accs = [it["mean_accuracy"] for it in items_sorted]
        threshold = byzantine_threshold(fracs, accs, cutoff=0.50)
        resil = resilience_score(fracs, accs)
        derived[combo_key] = {
            "honest_type": combo_key[0],
            "byzantine_type": combo_key[1],
            "committee_size": combo_key[2],
            "byzantine_threshold_50": round(threshold, 4),
            "resilience_score": round(resil, 4),
        }

    # Compute Byzantine amplification per (honest_type, committee_size)
    amplifications = []
    for honest in HONEST_NAMES:
        for n in COMMITTEE_SIZES:
            # Get accuracy at f=0.33 for strategic vs random
            strat_key = (honest, "strategic", n)
            rand_key = (honest, "random", n)
            if strat_key in derived and rand_key in derived:
                # Find accuracy at f=1/3 for each
                strat_items = combo_groups[strat_key]
                rand_items = combo_groups[rand_key]
                strat_acc_33 = next((it["mean_accuracy"] for it in strat_items if abs(it["byzantine_fraction"] - 1/3) < 0.01), None)
                rand_acc_33 = next((it["mean_accuracy"] for it in rand_items if abs(it["byzantine_fraction"] - 1/3) < 0.01), None)
                baseline_items = combo_groups.get((honest, "random", n), [])
                baseline_acc = next((it["mean_accuracy"] for it in baseline_items if it["byzantine_fraction"] == 0.0), None)
                if strat_acc_33 is not None and rand_acc_33 is not None and baseline_acc is not None:
                    amp = byzantine_amplification(strat_acc_33, rand_acc_33, baseline_acc)
                    amplifications.append({
                        "honest_type": honest,
                        "committee_size": n,
                        "amplification_at_f33": round(amp, 4),
                        "baseline_accuracy": round(baseline_acc, 6),
                        "strategic_accuracy_f33": round(strat_acc_33, 6),
                        "random_accuracy_f33": round(rand_acc_33, 6),
                    })

    return {
        "summaries": summaries,
        "derived_metrics": list(derived.values()),
        "amplifications": amplifications,
    }


def run_experiment(n_workers: int | None = None) -> dict[str, Any]:
    """Run the full experiment and return all results.

    Uses multiprocessing with *n_workers* parallel processes (defaults
    to cpu_count).
    """
    configs = _build_configs()
    n_workers = n_workers or max(1, cpu_count())

    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        raw_results = pool.map(_run_one, configs)
    elapsed = time.time() - t0

    aggregated = _aggregate_results(raw_results)

    return {
        "metadata": {
            "total_configs": len(configs),
            "honest_types": HONEST_NAMES,
            "byzantine_types": BYZANTINE_NAMES,
            "fractions": [round(f, 4) for f in FRACTIONS],
            "committee_sizes": COMMITTEE_SIZES,
            "seeds": SEEDS,
            "rounds_per_sim": ROUNDS_PER_SIM,
            "n_workers": n_workers,
            "elapsed_seconds": round(elapsed, 2),
        },
        "raw_results": raw_results,
        **aggregated,
    }


def save_results(results: dict[str, Any], out_dir: str = "results") -> Path:
    """Save results to JSON file."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / "results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    return out_file
