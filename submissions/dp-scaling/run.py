#!/usr/bin/env python3
"""Run the full DP scaling law experiment.

Usage (from submissions/dp-scaling/):
    .venv/bin/python run.py

Produces:
    results/experiment_results.json   -- Raw and aggregated results
    results/scaling_laws.png          -- Scaling law comparison figure
    results/accuracy_comparison.png   -- Accuracy comparison figure

Expected runtime: ~2-3 minutes on CPU.
"""

import os
import sys

# Working directory guard: ensure we run from the submission root
_script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != _script_dir:
    print(f"Changing directory to {_script_dir}")
    os.chdir(_script_dir)

sys.path.insert(0, _script_dir)

from src.experiment import run_full_experiment, save_results
from src.plot import plot_scaling_laws, plot_accuracy_comparison


def main() -> None:
    """Run the complete experiment pipeline."""
    print("=" * 60)
    print("DP Scaling Law Experiment")
    print("=" * 60)
    print()

    # Phase 1: Run experiments
    print("Phase 1: Running training experiments (45 runs)...")
    print("-" * 50)
    results = run_full_experiment()
    print()

    # Phase 2: Save results
    print("Phase 2: Saving results...")
    save_results(results, output_dir="results")
    print()

    # Phase 3: Generate plots
    print("Phase 3: Generating plots...")
    plot_scaling_laws(results, output_dir="results")
    plot_accuracy_comparison(results, output_dir="results")
    print()

    # Phase 4: Print summary
    print("=" * 60)
    print("SUMMARY: Scaling Law Exponents")
    print("=" * 60)
    for level, info in results["summary"].items():
        alpha = info.get("alpha")
        r2 = info.get("r_squared")
        ratio = info.get("alpha_ratio_vs_non_private")
        alpha_ci95 = info.get("alpha_ci95")
        if alpha is not None:
            print(f"  {level:15s}: alpha = {alpha:.4f}  (R^2 = {r2:.4f})", end="")
            if ratio is not None and level != "non_private":
                print(f"  ratio vs non-private = {ratio:.4f}", end="")
            print()
            if alpha_ci95 is not None:
                print(
                    " " * 19
                    + f"95% bootstrap CI: [{alpha_ci95[0]:.4f}, {alpha_ci95[1]:.4f}]"
                )
        else:
            print(f"  {level:15s}: fitting failed")

    print()
    print("Output files:")
    print("  results/experiment_results.json")
    print("  results/scaling_laws.png")
    print("  results/accuracy_comparison.png")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
