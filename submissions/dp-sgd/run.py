#!/usr/bin/env python3
"""Run the DP-SGD privacy-utility tradeoff experiment.

Sweeps noise multiplier sigma and clipping norm C across 3 seeds,
trains a 2-layer MLP on synthetic Gaussian cluster data, and
records accuracy and epsilon for each configuration.

Total: 7 sigma levels x 3 clipping norms x 3 seeds = 63 DP runs
       + 3 non-private baseline runs = 66 total runs.

Expected runtime: ~2 minutes on CPU.

Usage:
    .venv/bin/python run.py
"""

import json
import os
import sys
import time

# ── Working-directory guard ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import copy
import torch
from src.data import make_dataloaders
from src.model import create_model
from src.dpsgd import train_dpsgd, train_non_private
from src.analysis import (
    compute_summary_statistics,
    identify_privacy_cliff,
    generate_all_plots,
)

# ── Experiment Configuration ─────────────────────────────────────────
NOISE_MULTIPLIERS = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
CLIPPING_NORMS = [0.1, 1.0, 10.0]
SEEDS = [42, 123, 456]

N_SAMPLES = 500
N_FEATURES = 10
N_CLASSES = 5
CLUSTER_STD = 1.5
N_HIDDEN = 64
N_EPOCHS = 20
LEARNING_RATE = 0.1
BATCH_SIZE = 64
DELTA = 1e-5

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")


def main() -> None:
    """Run the full DP-SGD experiment sweep."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 60)
    print("DP-SGD Privacy-Utility Tradeoff Experiment")
    print("=" * 60)
    print(f"  Noise multipliers (sigma): {NOISE_MULTIPLIERS}")
    print(f"  Clipping norms (C):        {CLIPPING_NORMS}")
    print(f"  Seeds:                      {SEEDS}")
    print(f"  Total DP runs:              {len(NOISE_MULTIPLIERS) * len(CLIPPING_NORMS) * len(SEEDS)}")
    print(f"  Baseline runs:              {len(SEEDS)}")
    print(f"  Data: {N_SAMPLES} samples, {N_FEATURES} features, {N_CLASSES} classes")
    print(f"  Model: MLP (hidden={N_HIDDEN})")
    print(f"  Training: {N_EPOCHS} epochs, lr={LEARNING_RATE}, batch_size={BATCH_SIZE}")
    print(f"  Delta: {DELTA}")
    print("=" * 60)

    start_time = time.time()

    # ── Baseline (non-private) runs ──────────────────────────────────
    print("\n--- Non-private baseline ---")
    baseline_runs = []
    for seed in SEEDS:
        train_loader, test_loader, n_train, n_test = make_dataloaders(
            n_samples=N_SAMPLES, n_features=N_FEATURES, n_classes=N_CLASSES,
            cluster_std=CLUSTER_STD, seed=seed, batch_size=BATCH_SIZE,
        )
        model = create_model(
            n_features=N_FEATURES, n_hidden=N_HIDDEN,
            n_classes=N_CLASSES, seed=seed,
        )
        result = train_non_private(
            model=model, train_loader=train_loader, test_loader=test_loader,
            n_epochs=N_EPOCHS, learning_rate=LEARNING_RATE, seed=seed,
        )
        baseline_runs.append(result)
        print(f"  seed={seed}: accuracy={result['accuracy']:.4f}")

    baseline_mean = sum(r["accuracy"] for r in baseline_runs) / len(baseline_runs)
    print(f"  Baseline mean accuracy: {baseline_mean:.4f}")

    # ── DP-SGD sweep ─────────────────────────────────────────────────
    print("\n--- DP-SGD sweep ---")
    dp_runs = []
    total_configs = len(NOISE_MULTIPLIERS) * len(CLIPPING_NORMS) * len(SEEDS)
    run_idx = 0

    for sigma in NOISE_MULTIPLIERS:
        for C in CLIPPING_NORMS:
            for seed in SEEDS:
                run_idx += 1
                train_loader, test_loader, n_train, n_test = make_dataloaders(
                    n_samples=N_SAMPLES, n_features=N_FEATURES,
                    n_classes=N_CLASSES, cluster_std=CLUSTER_STD,
                    seed=seed, batch_size=BATCH_SIZE,
                )
                # Deep copy initial model to ensure identical initialization
                model = create_model(
                    n_features=N_FEATURES, n_hidden=N_HIDDEN,
                    n_classes=N_CLASSES, seed=seed,
                )

                result = train_dpsgd(
                    model=model, train_loader=train_loader,
                    test_loader=test_loader, n_epochs=N_EPOCHS,
                    learning_rate=LEARNING_RATE, max_norm=C,
                    noise_multiplier=sigma, n_train=n_train,
                    delta=DELTA, seed=seed,
                )
                dp_runs.append(result)

                utility_gap = result["accuracy"] - baseline_mean
                print(
                    f"  [{run_idx:3d}/{total_configs}] "
                    f"sigma={sigma:5.2f}, C={C:5.1f}, seed={seed:3d} -> "
                    f"acc={result['accuracy']:.4f}, "
                    f"eps={result['epsilon']:10.2f}, "
                    f"gap={utility_gap:+.4f}"
                )

    elapsed = time.time() - start_time
    print(f"\n--- Sweep complete in {elapsed:.1f}s ---")

    # ── Save raw results ─────────────────────────────────────────────
    all_results = {
        "config": {
            "noise_multipliers": NOISE_MULTIPLIERS,
            "clipping_norms": CLIPPING_NORMS,
            "seeds": SEEDS,
            "n_samples": N_SAMPLES,
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "cluster_std": CLUSTER_STD,
            "n_hidden": N_HIDDEN,
            "n_epochs": N_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "delta": DELTA,
        },
        "baseline_runs": baseline_runs,
        "dp_runs": dp_runs,
        "elapsed_seconds": elapsed,
    }

    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # ── Analysis & plots ─────────────────────────────────────────────
    print("\n--- Analysis ---")
    summaries = compute_summary_statistics(all_results)
    cliff = identify_privacy_cliff(summaries)

    print(f"  Baseline accuracy:   {summaries['baseline_accuracy_mean']:.4f} "
          f"(+/- {summaries['baseline_accuracy_std']:.4f})")

    if cliff["cliff_epsilon"] is not None:
        print(f"  Privacy cliff at:    epsilon ~ {cliff['cliff_epsilon']:.2f}")
    else:
        print("  Privacy cliff:       not detected (all configs above threshold)")

    if cliff["safe_epsilon"] is not None:
        print(f"  Safe region starts:  epsilon >= {cliff['safe_epsilon']:.2f} "
              f"(>=90% of baseline)")
    else:
        print("  Safe region:         no config reaches 90% of baseline")

    print(f"  Configs below 50%:   {cliff['n_configs_below_threshold']}/{cliff['n_configs_total']}")

    # ── Generate plots ───────────────────────────────────────────────
    print("\n--- Generating plots ---")
    plot_paths = generate_all_plots(RESULTS_DIR)
    for p in plot_paths:
        print(f"  Saved: {p}")

    # ── Key findings summary ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("KEY FINDINGS")
    print("=" * 60)

    # Find best DP config
    best_dp = max(dp_runs, key=lambda r: r["accuracy"])
    worst_dp = min(dp_runs, key=lambda r: r["accuracy"])

    print(f"  Best DP config:   sigma={best_dp['noise_multiplier']}, "
          f"C={best_dp['max_norm']}, acc={best_dp['accuracy']:.4f}, "
          f"eps={best_dp['epsilon']:.2f}")
    print(f"  Worst DP config:  sigma={worst_dp['noise_multiplier']}, "
          f"C={worst_dp['max_norm']}, acc={worst_dp['accuracy']:.4f}, "
          f"eps={worst_dp['epsilon']:.2f}")
    print(f"  Non-private:      acc={baseline_mean:.4f}")
    print(f"  Total runtime:    {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
