#!/usr/bin/env python3
"""Run the full adversarial world model experiment.

Usage:
    .venv/bin/python run.py [--n-rounds N] [--workers W] [--seeds S]

Defaults:
    --n-rounds 50000
    --workers   (cpu_count)
    --seeds     0,1,2
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from src.experiment import SimConfig, SimResult, build_experiment_matrix, run_simulation
from src.report import generate_full_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adversarial world model experiment")
    parser.add_argument("--n-rounds", type=int, default=50_000)
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--output-dir", type=str, default="results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Adversarial World Model Manipulation Experiment")
    print("=" * 60)
    print(f"  Rounds per sim:  {args.n_rounds:,}")
    print(f"  Seeds:           {seeds}")
    print(f"  Workers:         {args.workers}")
    print(f"  Output:          {output_dir}")

    # Build experiment matrix.
    configs = build_experiment_matrix(
        n_rounds=args.n_rounds,
        seeds=seeds,
    )
    print(f"  Total sims:      {len(configs)}")
    print()

    # Run simulations with multiprocessing.
    t0 = time.time()
    results: list[SimResult] = []

    with Pool(processes=args.workers) as pool:
        for i, result in enumerate(pool.imap_unordered(run_simulation, configs)):
            results.append(result)
            elapsed = time.time() - t0
            pct = (i + 1) / len(configs) * 100
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(configs) - i - 1) / rate if rate > 0 else 0
            sys.stdout.write(
                f"\r  [{i+1}/{len(configs)}] {pct:.0f}%  "
                f"{elapsed:.0f}s elapsed  ETA {eta:.0f}s  "
                f"({rate:.1f} sims/s)  "
            )
            sys.stdout.flush()

    total_time = time.time() - t0
    print(f"\n\nAll {len(results)} simulations completed in {total_time:.1f}s")

    # Sort results by config label for deterministic output.
    results.sort(key=lambda r: r.config.label)

    # Save raw results.
    raw_path = output_dir / "raw_results.pkl"
    with open(raw_path, "wb") as f:
        pickle.dump(results, f)
    print(f"Raw results saved to {raw_path}")

    # Generate report.
    print("\nGenerating report...")
    aggregated = generate_full_report(results, output_dir)

    # Print key findings.
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    for regime in ["stable", "volatile"]:
        print(f"\n--- {regime.upper()} (noise=0.0) ---")
        print(f"{'Matchup':<14} {'Belief Err':>10} {'Accuracy':>10}")
        print("-" * 36)
        for lc in ["NL", "SL", "AL"]:
            for ac in ["RA", "SA", "PA"]:
                key = f"{lc}-vs-{ac}_{regime}_noise0.0"
                if key in aggregated:
                    be = aggregated[key].get("distortion.final_belief_error.mean", float("nan"))
                    acc = aggregated[key].get("decision_quality.accuracy.mean", float("nan"))
                    print(f"{lc}-vs-{ac:<8} {be:>10.4f} {acc:>10.4f}")

    print(f"\nTotal runtime: {total_time:.1f}s")
    print("DONE")


if __name__ == "__main__":
    main()
