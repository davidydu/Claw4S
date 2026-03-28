#!/usr/bin/env python3
"""Run the full label-noise tolerance experiment.

Usage (from submissions/label-noise/):
    .venv/bin/python run.py

Outputs:
    results/raw_results.json   — per-run metrics (168 rows)
    results/summary.json       — aggregated mean +/- std
    results/arch_sweep.png     — architecture comparison plots
    results/width_sweep.png    — width sweep plots
"""

import os
import sys
import time


def main() -> None:
    # ---- working-directory guard ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sys.path.insert(0, script_dir)

    from src.experiment import run_all
    from src.plot import plot_all

    print("=" * 60)
    print("Label Noise Tolerance Curves — Full Experiment")
    print("=" * 60)
    t0 = time.time()

    summary = run_all(results_dir="results")

    print()
    print("Generating plots...")
    paths = plot_all(results_dir="results")
    for p in paths:
        print(f"  -> {p}")

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Done in {elapsed:.1f}s")
    print("=" * 60)

    # Print key findings
    print()
    print("KEY FINDINGS:")
    for i, finding in enumerate(summary.get("findings", []), 1):
        print(f"  {i}. {finding}")


if __name__ == "__main__":
    main()
