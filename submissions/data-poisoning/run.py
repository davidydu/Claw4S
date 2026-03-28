"""Run the full data poisoning sensitivity experiment.

Usage: .venv/bin/python run.py
Must be run from the submissions/data-poisoning/ directory.
"""

import json
import os
import sys
import time

# ── Working-directory guard ──────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != _here:
    print(f"Changing working directory to {_here}")
    os.chdir(_here)

sys.path.insert(0, _here)

from src.experiment import ExperimentConfig, run_sweep
from src.analysis import (
    aggregate_results,
    build_performance_payload,
    build_results_payload,
    compute_findings,
    fit_sigmoid_curve,
)
from src.plotting import plot_accuracy_vs_poison, plot_generalization_gap, plot_train_vs_test


def main() -> None:
    """Run the full experiment pipeline."""
    print("=" * 60)
    print("Data Poisoning Sensitivity Experiment")
    print("=" * 60)

    config = ExperimentConfig()
    total_runs = (
        len(config.poison_fractions)
        * len(config.hidden_widths)
        * len(config.seeds)
    )
    print(f"\nConfig: {len(config.poison_fractions)} poison fractions "
          f"x {len(config.hidden_widths)} model sizes "
          f"x {len(config.seeds)} seeds = {total_runs} runs")

    # ── Step 1: Run sweep ────────────────────────────────────────────
    print("\n[Step 1/4] Running poisoning sweep...")
    t0 = time.time()
    results = run_sweep(config)
    elapsed = time.time() - t0
    print(f"  Completed {len(results)} runs in {elapsed:.1f}s")

    # ── Step 2: Aggregate ────────────────────────────────────────────
    print("\n[Step 2/4] Aggregating results...")
    agg = aggregate_results(results)
    print(f"  {len(agg)} aggregated data points")

    # ── Step 3: Fit sigmoid curves ───────────────────────────────────
    print("\n[Step 3/4] Fitting sigmoid curves...")
    fits = []
    for hw in config.hidden_widths:
        try:
            fit = fit_sigmoid_curve(agg, hw)
            fits.append(fit)
            print(f"  Width {hw}: k={fit.k:.2f}, x0={fit.x0:.3f}, "
                  f"threshold_mid={fit.threshold_midpoint:.3f}, R²={fit.r_squared:.4f}")
        except RuntimeError as exc:
            print(f"  Width {hw}: fit failed - {exc}")

    findings = compute_findings(agg, fits)

    # ── Step 4: Generate plots and save results ──────────────────────
    print("\n[Step 4/4] Generating plots and saving results...")
    os.makedirs("results", exist_ok=True)

    plot_accuracy_vs_poison(agg, fits, "results")
    plot_generalization_gap(agg, "results")
    plot_train_vs_test(agg, "results")
    print("  Saved: accuracy_vs_poison.png, generalization_gap.png, train_vs_test.png")

    # Save JSON results
    output = build_results_payload(config, results, agg, fits, findings)
    with open("results/results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print("  Saved: results.json")

    performance = build_performance_payload(results, elapsed)
    with open("results/performance.json", "w") as f:
        json.dump(performance, f, indent=2, default=str)
    print("  Saved: performance.json")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)
    print(f"\nClean test accuracy (no poison):")
    for hw, acc in findings.get("clean_test_accuracy", {}).items():
        print(f"  Width {hw}: {acc:.3f}")

    print(f"\nCritical thresholds (midpoint between clean and chance):")
    for hw, thresh in findings.get("critical_thresholds", {}).items():
        if thresh == float("inf"):
            print(f"  Width {hw}: > 50% poison (never reached)")
        else:
            print(f"  Width {hw}: {thresh:.1%} poison")

    print(f"\nSigmoid steepness (k, higher = sharper transition):")
    for hw, k in findings.get("steepness_k", {}).items():
        print(f"  Width {hw}: k={k:.2f}")

    if findings.get("sharp_transition"):
        print("\nPhase transition: SHARP (k > 5 for at least one model)")
    else:
        print("\nPhase transition: GRADUAL (k <= 5 for all models)")

    sensitive = findings.get("larger_models_more_sensitive")
    if sensitive is True:
        print("Larger models: MORE SENSITIVE to poisoning (lower threshold)")
    elif sensitive is False:
        print("Larger models: NOT more sensitive to poisoning")
    else:
        print("Larger models: sensitivity comparison inconclusive")

    print(f"\nGeneralization gap at 50% poison:")
    for hw, gap in findings.get("gen_gap_at_50pct_poison", {}).items():
        print(f"  Width {hw}: {gap:.3f}")

    print(f"\nTotal experiment time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
