"""Run the full double descent analysis and generate report.

Must be run from the submissions/double-descent/ directory.
"""

import json
import os
import sys
import traceback

# Working-directory guard
expected_marker = os.path.join("src", "sweep.py")
if not os.path.exists(expected_marker):
    print(
        "ERROR: run.py must be run from the submissions/double-descent/ directory.\n"
        "  cd submissions/double-descent/ && .venv/bin/python run.py"
    )
    sys.exit(1)

try:
    from src.sweep import run_all_sweeps
    from src.analysis import compute_variance_bands
    from src.plots import (
        plot_model_wise,
        plot_noise_comparison,
        plot_epoch_wise,
        plot_mlp_comparison,
        plot_variance_bands,
    )
    from src.report import generate_report

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Run all experiments
    print("=" * 60)
    print("Double Descent in Practice — Full Analysis")
    print("=" * 60)
    print()

    all_results = run_all_sweeps()
    meta = all_results["metadata"]

    # Generate plots
    print()
    print("[5/6] Generating plots...")

    threshold = meta["rf_interpolation_threshold"]
    highest_noise = f"noise_{max(meta['noise_levels'])}"

    plot_model_wise(
        all_results["random_features"],
        interpolation_threshold=threshold,
        output_path="results/model_wise_double_descent.png",
    )
    print("  Saved results/model_wise_double_descent.png")

    plot_noise_comparison(
        all_results["random_features"],
        interpolation_threshold=threshold,
        output_path="results/noise_comparison.png",
    )
    print("  Saved results/noise_comparison.png")

    plot_epoch_wise(
        all_results["epoch_wise"],
        output_path="results/epoch_wise_double_descent.png",
    )
    print("  Saved results/epoch_wise_double_descent.png")

    plot_mlp_comparison(
        all_results["mlp_sweep"],
        all_results["random_features"][highest_noise],
        n_train=meta["n_train"],
        mlp_interpolation_threshold=meta["mlp_interpolation_threshold"],
        output_path="results/mlp_comparison.png",
    )
    print("  Saved results/mlp_comparison.png")

    variance_stats = compute_variance_bands(all_results["variance"])
    plot_variance_bands(
        variance_stats,
        interpolation_threshold=threshold,
        output_path="results/variance_bands.png",
    )
    print("  Saved results/variance_bands.png")

    # Generate report
    print()
    print("[6/6] Saving results...")

    report = generate_report(all_results)
    with open("results/report.md", "w") as f:
        f.write(report)
    print("  Saved results/report.md")

    with open("results/results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("  Saved results/results.json")

    print()
    print(f"Done. Total runtime: {meta['runtime_seconds']:.1f}s")
    print()
    print(report)

except Exception:
    traceback.print_exc()
    sys.exit(1)
