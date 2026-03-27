"""Run the full DP noise calibration comparison analysis.

Computes privacy loss (epsilon) across a grid of noise multipliers,
composition steps, and failure probabilities using four accounting
methods: naive composition, advanced composition, Renyi DP, and
Gaussian DP. Saves results and generates visualizations.

Must be run from the submission directory:
    .venv/bin/python run.py
"""

import os
import sys

# --- Working-directory guard ---
_expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "SKILL.md")
if not os.path.isfile(_expected_marker):
    print("ERROR: run.py must be executed from the dp-calibration/ "
          "submission directory.", file=sys.stderr)
    sys.exit(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --------------------------------

from src.analysis import run_analysis, save_results
from src.visualize import generate_all_figures


def main():
    print("=" * 60)
    print("DP Noise Calibration Comparison")
    print("=" * 60)
    print()

    # Step 1: Run the parameter sweep
    print("[1/3] Running parameter sweep across (T, delta, sigma) grid...")
    data = run_analysis(seed=42)

    meta = data["metadata"]
    print(f"      Grid: {meta['num_T']} T values x "
          f"{meta['num_delta']} delta values x "
          f"{meta['num_sigma']} sigma values")
    print(f"      Methods: {meta['num_methods']}")
    print(f"      Total computations: {meta['total_computations']}")
    print(f"      Elapsed: {meta['elapsed_seconds']:.3f}s")
    print()

    # Step 2: Save results
    print("[2/3] Saving results to results/results.json...")
    path = save_results(data)
    print(f"      Saved to: {path}")
    print()

    # Step 3: Generate visualizations
    print("[3/3] Generating visualizations...")
    fig_paths = generate_all_figures(data)
    for p in fig_paths:
        print(f"      Saved: {p}")
    print()

    # Summary
    summary = data["summary"]
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print("Method Win Counts (tightest bound):")
    for method, count in summary["win_counts"].items():
        pct = 100 * count / meta["total_configs"] if meta["total_configs"] > 0 else 0
        print(f"  {method:12s}: {count:3d} / {meta['total_configs']} ({pct:.1f}%)")
    print()
    print("Average Tightness Ratio (lower = tighter, 1.0 = optimal):")
    for method in summary["avg_tightness_ratio"]:
        avg = summary["avg_tightness_ratio"][method]
        std = summary["std_tightness_ratio"][method]
        if avg == float("inf") or avg == "Infinity":
            print(f"  {method:12s}: inf")
        else:
            print(f"  {method:12s}: {avg:.4f} +/- {std:.4f}")
    print()
    print("Wins by Composition Steps (T):")
    for T_str, wins in summary["wins_by_T"].items():
        winners = [f"{m}={c}" for m, c in wins.items() if c > 0]
        print(f"  T={T_str:>5s}: {', '.join(winners)}")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
