"""Run the full emergent abilities analysis and generate all outputs.

This script orchestrates:
1. Core analysis (metric comparisons, nonlinearity detection, synthetic demo, MMLU)
2. Figure generation
3. Report generation
4. Results serialization to JSON

All outputs are written to results/ directory.
"""

import json
import os
import sys
import traceback

# Working-directory guard: must run from submission directory
if not os.path.isfile("run.py"):
    print("ERROR: run.py must be executed from the submission directory.")
    print("  cd submissions/emergent-abilities/")
    sys.exit(1)

os.makedirs("results/figures", exist_ok=True)

from src.analysis import run_full_analysis
from src.plots import (
    plot_metric_comparison,
    plot_synthetic_demo,
    plot_nonlinearity_heatmap,
    plot_mmlu_scaling,
)
from src.report import generate_report

SEED = 42


def main() -> None:
    print("=" * 60)
    print("Emergent Abilities Analysis: Mirage or Real?")
    print("=" * 60)

    # Step 1: Run all analyses
    print("\n[1/4] Running analyses...")
    try:
        results = run_full_analysis(seed=SEED)
    except Exception as e:
        print(f"ERROR in analysis: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    n_tasks = len(results["metric_comparisons"])
    n_scores = len(results["nonlinearity_scores"])
    print(f"  Analyzed {n_tasks} BIG-Bench tasks")
    print(f"  Computed nonlinearity scores for {n_scores} tasks")
    print(f"  MMLU: {results['mmlu_analysis']['n_models']} models analyzed")

    # Step 2: Generate figures
    print("\n[2/4] Generating figures...")
    try:
        # Metric comparison plots for key tasks
        for task_name in ["2_digit_multiplication", "4_digit_addition", "ipa_transliterate"]:
            if task_name in results["metric_comparisons"]:
                outpath = f"results/figures/metric_comparison_{task_name}.png"
                plot_metric_comparison(results["metric_comparisons"][task_name], outpath)
                print(f"  Saved {outpath}")

        # Synthetic demo
        plot_synthetic_demo(results["synthetic_demo"], "results/figures/synthetic_demo.png")
        print("  Saved results/figures/synthetic_demo.png")

        # Nonlinearity heatmap
        plot_nonlinearity_heatmap(
            results["nonlinearity_scores"], "results/figures/nonlinearity_heatmap.png"
        )
        print("  Saved results/figures/nonlinearity_heatmap.png")

        # MMLU scaling
        plot_mmlu_scaling(results["mmlu_analysis"], "results/figures/mmlu_scaling.png")
        print("  Saved results/figures/mmlu_scaling.png")

    except Exception as e:
        print(f"ERROR generating figures: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Generate report
    print("\n[3/4] Generating report...")
    try:
        report = generate_report(results)
        with open("results/report.md", "w") as f:
            f.write(report)
        print("  Saved results/report.md")
    except Exception as e:
        print(f"ERROR generating report: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Step 4: Save JSON results
    print("\n[4/4] Saving results to results/")
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_results = _make_serializable(results)
        with open("results/results.json", "w") as f:
            json.dump(json_results, f, indent=2)
        print("  Saved results/results.json")
    except Exception as e:
        print(f"ERROR saving results: {type(e).__name__}: {e}")
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Analysis complete. See results/ for outputs.")
    print("=" * 60)


def _make_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON."""
    import numpy as np

    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, float) and (obj != obj):  # NaN check
        return None
    elif isinstance(obj, float) and abs(obj) == float("inf"):
        return str(obj)
    else:
        return obj


if __name__ == "__main__":
    main()
