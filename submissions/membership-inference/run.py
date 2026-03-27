#!/usr/bin/env python3
"""Main runner for membership inference scaling experiment.

Must be run from submissions/membership-inference/ directory.
Trains target and shadow models across 5 MLP widths, runs membership
inference attacks, computes correlations, and generates plots + report.
"""

import os
import sys
import time


def check_working_directory() -> None:
    """Verify we are running from the correct directory."""
    if not os.path.isfile("SKILL.md"):
        print(
            "ERROR: run.py must be run from submissions/membership-inference/",
            file=sys.stderr,
        )
        print(
            f"Current directory: {os.getcwd()}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    check_working_directory()

    start_time = time.time()

    # Import after path check to give clear error if deps are missing
    from src.data import generate_gaussian_clusters, split_data, SEED
    from src.models import HIDDEN_WIDTHS
    from src.attack import run_attack_for_width
    from src.analysis import (
        compute_correlations,
        generate_plots,
        generate_report,
        save_results,
    )

    print("=" * 60)
    print("Membership Inference Scaling Experiment")
    print("=" * 60)

    # Step 1: Generate target data
    print("\n[1/4] Generating synthetic classification data...")
    X, y = generate_gaussian_clusters(seed=SEED)
    X_train, y_train, X_test, y_test = split_data(X, y, seed=SEED)
    print(f"  Data: {len(X)} samples, {X.shape[1]} features, "
          f"train={len(X_train)}, test={len(X_test)}")

    # Step 2: Run attacks for each model width
    print("\n[2/4] Running membership inference attacks...")
    all_results = []
    for width in HIDDEN_WIDTHS:
        print(f"  Width={width}...", end=" ", flush=True)
        width_start = time.time()
        result = run_attack_for_width(
            hidden_width=width,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            seed=SEED,
        )
        elapsed = time.time() - width_start
        print(
            f"AUC={result['mean_attack_auc']:.3f} +/- {result['std_attack_auc']:.3f}, "
            f"Gap={result['mean_overfit_gap']:.3f} ({elapsed:.1f}s)"
        )
        all_results.append(result)

    # Step 3: Statistical analysis
    print("\n[3/4] Computing correlations and generating plots...")
    correlations = compute_correlations(all_results)

    for key, corr in correlations.items():
        sig = "*" if corr["p"] < 0.05 else ""
        print(f"  {corr['description']}: r={corr['r']:.4f}, p={corr['p']:.4f}{sig}")

    plots = generate_plots(all_results, "results/")
    if plots:
        print(f"  Saved {len(plots)} plots to results/")
    else:
        print("  (matplotlib not available, skipping plots)")

    # Step 4: Save results
    print("\n[4/4] Saving results to results/")
    save_results(all_results, correlations, "results/results.json")
    generate_report(all_results, correlations, "results/report.md")

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f}s")
    print(f"  results/results.json  - raw data")
    print(f"  results/report.md     - summary report")


if __name__ == "__main__":
    main()
