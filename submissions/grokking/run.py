"""Run the full grokking phase diagram analysis.

This script orchestrates:
  1. Phase diagram sweep across weight_decay x dataset_fraction x hidden_dim
  2. Phase classification for each run
  3. Visualization (phase diagram heatmaps + training curves)
  4. Report generation

All results are saved to the results/ directory.
"""

import json
import os
import sys
import traceback

# Ensure we're running from the submission directory
if not os.path.isfile("run.py"):
    print("ERROR: run.py must be executed from the submissions/grokking/ directory.")
    print(f"  Current directory: {os.getcwd()}")
    sys.exit(1)


def main() -> None:
    """Run the full analysis pipeline."""
    import numpy as np
    import torch

    from src.sweep import run_sweep
    from src.plots import plot_phase_diagram, plot_grokking_curves
    from src.report import generate_report
    from src.analysis import aggregate_results, Phase

    # Pin seeds globally
    torch.manual_seed(42)
    np.random.seed(42)

    os.makedirs("results", exist_ok=True)

    # Step 1: Run the sweep
    print("[1/4] Running phase diagram sweep...")
    print("=" * 60)
    results = run_sweep()
    print()

    # Step 2: Save raw results
    print("[2/4] Saving raw results...")
    with open("results/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results to results/sweep_results.json")

    # Save phase diagram summary
    results_for_agg = []
    for r in results:
        phase = r["phase"]
        if isinstance(phase, str):
            phase_enum = Phase(phase)
        else:
            phase_enum = phase
        results_for_agg.append({
            "phase": phase_enum,
            "grokking_gap": r.get("grokking_gap"),
        })

    stats = aggregate_results(results_for_agg)
    phase_summary = {
        "phase_counts": stats["phase_counts"],
        "total_runs": stats["total_runs"],
        "grokking_fraction": stats["grokking_fraction"],
        "mean_grokking_gap": stats["mean_grokking_gap"],
        "max_grokking_gap": stats["max_grokking_gap"],
    }
    with open("results/phase_diagram.json", "w") as f:
        json.dump(phase_summary, f, indent=2)
    print("  Saved phase summary to results/phase_diagram.json")
    print()

    # Step 3: Generate plots
    print("[3/4] Generating plots...")
    hidden_dims = sorted(set(r["config"]["hidden_dim"] for r in results))
    for hd in hidden_dims:
        plot_phase_diagram(results, hd, f"results/phase_diagram_h{hd}.png")
    plot_grokking_curves(results, "results/grokking_curves.png")
    print()

    # Step 4: Generate report
    print("[4/4] Generating report...")
    report = generate_report(results)
    with open("results/report.md", "w") as f:
        f.write(report)
    print("  Saved report to results/report.md")
    print()

    # Summary
    print("=" * 60)
    print("Analysis complete!")
    print(f"  Total runs: {stats['total_runs']}")
    for phase in Phase:
        count = stats['phase_counts'].get(phase.value, 0)
        print(f"  {phase.value.capitalize()}: {count}")
    if stats["mean_grokking_gap"] is not None:
        print(f"  Mean grokking gap: {stats['mean_grokking_gap']:.0f} epochs")
    print()
    print("Results saved to results/")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)
