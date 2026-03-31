"""Main runner: execute full adversarial transferability experiment.

Usage (from submissions/transferability/):
    .venv/bin/python run.py

Outputs:
    results/transfer_results.json   -- raw data + summary statistics
    results/transfer_heatmap.png    -- 4x4 transfer rate heatmap
    results/transfer_by_ratio.png   -- transfer rate vs capacity ratio
    results/depth_comparison.png    -- same-depth vs cross-depth comparison
"""

import os
import sys

# Working-directory guard: ensure we run from the submission root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(_SCRIPT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from pathlib import Path
from src.experiment import run_full_experiment
from src.visualize import generate_all_plots


def main() -> None:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("Adversarial Transferability Phase Diagram")
    print("=" * 60)

    # Phase 1+2: Run experiments
    output = run_full_experiment(results_dir)

    # Phase 3: Generate plots
    print("\n" + "=" * 60)
    print("PHASE 3: Generating visualizations")
    print("=" * 60)
    generate_all_plots(results_dir)

    # Phase 4: Print summary
    summary = output["summary"]
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Same-arch runs: {summary['n_same_arch_runs']}")
    print(f"Cross-depth runs: {summary['n_cross_depth_runs']}")
    print(f"Runtime: {summary['runtime_seconds']}s")
    print(f"\nDiagonal (same-width) mean transfer: {summary['diagonal_mean_transfer']}")
    print(f"Off-diagonal mean transfer: {summary['off_diagonal_mean_transfer']}")
    print(f"\nTransfer by capacity ratio:")
    for ratio, mean in summary["transfer_by_capacity_ratio"].items():
        print(f"  ratio={ratio}: {mean}")
    print(f"\nSame-width same-depth mean: {summary['same_width_same_depth_mean']}")
    print(f"Same-width cross-depth mean: {summary['same_width_cross_depth_mean']}")
    print("\nDone. Check results/ for full output.")


if __name__ == "__main__":
    main()
