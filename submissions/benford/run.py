"""Run the full Benford's Law analysis on trained neural networks.

Trains tiny MLPs on modular arithmetic and sine regression tasks,
saves weight snapshots across training, extracts leading digit distributions,
and tests conformity to Benford's Law using chi-squared and MAD statistics.
"""

import json
import os
import sys
import time

# Working directory guard
if not os.path.exists("src/benford_analysis.py"):
    print("ERROR: Must run from submissions/benford/ directory.", file=sys.stderr)
    sys.exit(1)

import torch

from src.benford_analysis import analyze_snapshot, generate_control_weights
from src.data import generate_modular_data, generate_sine_data
from src.model import TinyMLP
from src.plots import (
    plot_controls_comparison,
    plot_digit_distribution,
    plot_layer_comparison,
    plot_mad_over_training,
)
from src.report import generate_report
from src.train import set_seed, train_model

# Configuration
SEED = 42
HIDDEN_SIZES = [64, 128]
SNAPSHOT_EPOCHS = [0, 100, 500, 1000, 2000, 5000]
MOD_P = 97
SINE_N = 1000
RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def main():
    start_time = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)

    all_results = {
        "models": {},
        "controls": {},
        "metadata": {
            "seed": SEED,
            "tasks": ["modular_arithmetic", "sine_regression"],
            "hidden_sizes": HIDDEN_SIZES,
            "snapshot_epochs": SNAPSHOT_EPOCHS,
            "mod_p": MOD_P,
            "sine_n": SINE_N,
        },
    }

    # --- Task 1: Modular Arithmetic ---
    print("[1/4] Generating modular arithmetic data (mod {})...".format(MOD_P))
    X_train_mod, y_train_mod, X_test_mod, y_test_mod = generate_modular_data(
        p=MOD_P, seed=SEED
    )

    for h in HIDDEN_SIZES:
        model_name = f"mod{MOD_P}_h{h}"
        print(f"  Training {model_name}...")

        set_seed(SEED)  # Ensure deterministic model init
        model = TinyMLP(d_in=2, d_hidden=h, d_out=MOD_P, n_hidden=2)
        print(f"    Parameters: {model.count_parameters()}")

        snapshots, history = train_model(
            model=model,
            X_train=X_train_mod,
            y_train=y_train_mod,
            task_type="classification",
            epochs=5000,
            lr=1e-3,
            snapshot_epochs=SNAPSHOT_EPOCHS,
            seed=SEED,
            X_test=X_test_mod,
            y_test=y_test_mod,
        )

        print(f"    Final train loss: {history['train_loss'][-1]:.4f}")
        if history["test_loss"]:
            print(f"    Final test loss: {history['test_loss'][-1]:.4f}")

        # Analyze each snapshot
        epoch_results = {}
        for epoch, state_dict in sorted(snapshots.items()):
            analysis = analyze_snapshot(state_dict)
            epoch_results[str(epoch)] = analysis

        all_results["models"][model_name] = epoch_results

        # Plot final epoch digit distribution
        final_epoch = str(max(snapshots.keys()))
        final_agg = epoch_results[final_epoch]["aggregate"]
        plot_digit_distribution(
            final_agg["observed_dist"],
            f"{model_name} - Epoch {final_epoch} (Aggregate)",
            os.path.join(FIGURES_DIR, f"{model_name}_digits.png"),
        )

        # Plot MAD over training
        mad_by_epoch = {
            int(e): r["aggregate"] for e, r in epoch_results.items()
        }
        plot_mad_over_training(
            mad_by_epoch,
            f"{model_name} - MAD over Training",
            os.path.join(FIGURES_DIR, f"{model_name}_mad_training.png"),
        )

        # Plot per-layer comparison at final epoch
        final_layers = epoch_results[final_epoch]["per_layer"]
        if final_layers:
            plot_layer_comparison(
                final_layers,
                f"{model_name} - Per-Layer MAD (Epoch {final_epoch})",
                os.path.join(FIGURES_DIR, f"{model_name}_layers.png"),
            )

    # --- Task 2: Sine Regression ---
    print("[2/4] Generating sine regression data...")
    X_train_sin, y_train_sin, X_test_sin, y_test_sin = generate_sine_data(
        n=SINE_N, seed=SEED
    )

    for h in HIDDEN_SIZES:
        model_name = f"sine_h{h}"
        print(f"  Training {model_name}...")

        set_seed(SEED)  # Ensure deterministic model init
        model = TinyMLP(d_in=1, d_hidden=h, d_out=1, n_hidden=2)
        print(f"    Parameters: {model.count_parameters()}")

        snapshots, history = train_model(
            model=model,
            X_train=X_train_sin,
            y_train=y_train_sin,
            task_type="regression",
            epochs=5000,
            lr=1e-3,
            snapshot_epochs=SNAPSHOT_EPOCHS,
            seed=SEED,
            X_test=X_test_sin,
            y_test=y_test_sin,
        )

        print(f"    Final train loss: {history['train_loss'][-1]:.6f}")
        if history["test_loss"]:
            print(f"    Final test loss: {history['test_loss'][-1]:.6f}")

        epoch_results = {}
        for epoch, state_dict in sorted(snapshots.items()):
            analysis = analyze_snapshot(state_dict)
            epoch_results[str(epoch)] = analysis

        all_results["models"][model_name] = epoch_results

        final_epoch = str(max(snapshots.keys()))
        final_agg = epoch_results[final_epoch]["aggregate"]
        plot_digit_distribution(
            final_agg["observed_dist"],
            f"{model_name} - Epoch {final_epoch} (Aggregate)",
            os.path.join(FIGURES_DIR, f"{model_name}_digits.png"),
        )

        mad_by_epoch = {
            int(e): r["aggregate"] for e, r in epoch_results.items()
        }
        plot_mad_over_training(
            mad_by_epoch,
            f"{model_name} - MAD over Training",
            os.path.join(FIGURES_DIR, f"{model_name}_mad_training.png"),
        )

        final_layers = epoch_results[final_epoch]["per_layer"]
        if final_layers:
            plot_layer_comparison(
                final_layers,
                f"{model_name} - Per-Layer MAD (Epoch {final_epoch})",
                os.path.join(FIGURES_DIR, f"{model_name}_layers.png"),
            )

    # --- Controls ---
    print("[3/4] Generating control distributions...")
    controls = generate_control_weights(n=10000, seed=SEED)
    all_results["controls"] = controls

    plot_controls_comparison(
        controls, os.path.join(FIGURES_DIR, "controls_comparison.png")
    )

    # --- Save results ---
    elapsed = time.time() - start_time
    all_results["metadata"]["runtime_seconds"] = elapsed

    print("[4/4] Saving results to results/")
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    report = generate_report(all_results)
    with open(os.path.join(RESULTS_DIR, "report.md"), "w") as f:
        f.write(report)

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Results: {RESULTS_DIR}/results.json")
    print(f"  Report:  {RESULTS_DIR}/report.md")
    print(f"  Figures: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
