"""Run the full Benford's Law analysis on trained neural networks.

Trains tiny MLPs on modular arithmetic and sine regression tasks,
saves weight snapshots across training, extracts leading digit distributions,
and tests conformity to Benford's Law using chi-squared and MAD statistics.
"""

import argparse
import json
import os
import platform
import sys
import time

# Working directory guard
if not os.path.exists("src/benford_analysis.py"):
    print("ERROR: Must run from submissions/benford/ directory.", file=sys.stderr)
    sys.exit(1)

import matplotlib
import numpy as np
import scipy
import torch

from src.benford_analysis import analyze_snapshot, generate_control_weights
from src.config import parse_int_list, resolve_run_config
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

RESULTS_DIR = "results"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def parse_args(argv=None):
    """Parse command-line args and resolve run configuration."""
    parser = argparse.ArgumentParser(
        description=(
            "Benford's Law analysis of trained tiny MLPs. "
            "Defaults reproduce the full submission run."
        )
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a faster smoke-test profile (smaller sweep, fewer epochs).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default=None,
        help="Comma-separated hidden sizes, e.g. '64,128'.",
    )
    parser.add_argument(
        "--snapshot-epochs",
        type=str,
        default=None,
        help="Comma-separated snapshot epochs, e.g. '0,100,500,5000'.",
    )
    parser.add_argument("--mod-p", type=int, default=None, help="Modulo for arithmetic task.")
    parser.add_argument("--sine-n", type=int, default=None, help="Number of sine samples.")
    parser.add_argument("--epochs", type=int, default=None, help="Training epochs per model.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument(
        "--controls-n", type=int, default=None, help="Number of samples per control distribution."
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=None,
        help="Progress logging interval in epochs (default depends on profile).",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip figure generation (faster, still writes JSON + report).",
    )
    args = parser.parse_args(argv)

    hidden_sizes = (
        parse_int_list(args.hidden_sizes, field_name="hidden_sizes")
        if args.hidden_sizes
        else None
    )
    snapshot_epochs = (
        parse_int_list(args.snapshot_epochs, field_name="snapshot_epochs")
        if args.snapshot_epochs
        else None
    )

    return resolve_run_config(
        quick=args.quick,
        seed=args.seed,
        hidden_sizes=hidden_sizes,
        snapshot_epochs=snapshot_epochs,
        mod_p=args.mod_p,
        sine_n=args.sine_n,
        epochs=args.epochs,
        lr=args.lr,
        controls_n=args.controls_n,
        log_every=args.log_every,
        make_plots=not args.skip_plots,
    )


def _build_metadata(config):
    """Create reproducibility-focused run metadata."""
    return {
        "seed": config.seed,
        "tasks": ["modular_arithmetic", "sine_regression"],
        "hidden_sizes": config.hidden_sizes,
        "snapshot_epochs": config.snapshot_epochs,
        "mod_p": config.mod_p,
        "sine_n": config.sine_n,
        "epochs": config.epochs,
        "learning_rate": config.lr,
        "controls_n": config.controls_n,
        "quick_mode": config.quick,
        "make_plots": config.make_plots,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "matplotlib_version": matplotlib.__version__,
    }


def main(config):
    start_time = time.time()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    print(
        "Resolved config: "
        f"quick={config.quick}, epochs={config.epochs}, hidden_sizes={config.hidden_sizes}, "
        f"snapshot_epochs={config.snapshot_epochs}, controls_n={config.controls_n}, "
        f"plots={'on' if config.make_plots else 'off'}"
    )

    all_results = {
        "models": {},
        "controls": {},
        "metadata": _build_metadata(config),
    }

    # --- Task 1: Modular Arithmetic ---
    print("[1/4] Generating modular arithmetic data (mod {})...".format(config.mod_p))
    X_train_mod, y_train_mod, X_test_mod, y_test_mod = generate_modular_data(
        p=config.mod_p, seed=config.seed
    )

    for h in config.hidden_sizes:
        model_name = f"mod{config.mod_p}_h{h}"
        print(f"  Training {model_name}...")

        set_seed(config.seed)  # Ensure deterministic model init
        model = TinyMLP(d_in=2, d_hidden=h, d_out=config.mod_p, n_hidden=2)
        print(f"    Parameters: {model.count_parameters()}")

        snapshots, history = train_model(
            model=model,
            X_train=X_train_mod,
            y_train=y_train_mod,
            task_type="classification",
            epochs=config.epochs,
            lr=config.lr,
            snapshot_epochs=config.snapshot_epochs,
            seed=config.seed,
            X_test=X_test_mod,
            y_test=y_test_mod,
            log_every=config.log_every,
            log_prefix=f"    [{model_name}]",
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

        if config.make_plots:
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
            final_layers = epoch_results[str(max(snapshots.keys()))]["per_layer"]
            if final_layers:
                plot_layer_comparison(
                    final_layers,
                    f"{model_name} - Per-Layer MAD (Epoch {max(snapshots.keys())})",
                    os.path.join(FIGURES_DIR, f"{model_name}_layers.png"),
                )

    # --- Task 2: Sine Regression ---
    print("[2/4] Generating sine regression data...")
    X_train_sin, y_train_sin, X_test_sin, y_test_sin = generate_sine_data(
        n=config.sine_n, seed=config.seed
    )

    for h in config.hidden_sizes:
        model_name = f"sine_h{h}"
        print(f"  Training {model_name}...")

        set_seed(config.seed)  # Ensure deterministic model init
        model = TinyMLP(d_in=1, d_hidden=h, d_out=1, n_hidden=2)
        print(f"    Parameters: {model.count_parameters()}")

        snapshots, history = train_model(
            model=model,
            X_train=X_train_sin,
            y_train=y_train_sin,
            task_type="regression",
            epochs=config.epochs,
            lr=config.lr,
            snapshot_epochs=config.snapshot_epochs,
            seed=config.seed,
            X_test=X_test_sin,
            y_test=y_test_sin,
            log_every=config.log_every,
            log_prefix=f"    [{model_name}]",
        )

        print(f"    Final train loss: {history['train_loss'][-1]:.6f}")
        if history["test_loss"]:
            print(f"    Final test loss: {history['test_loss'][-1]:.6f}")

        epoch_results = {}
        for epoch, state_dict in sorted(snapshots.items()):
            analysis = analyze_snapshot(state_dict)
            epoch_results[str(epoch)] = analysis

        all_results["models"][model_name] = epoch_results

        if config.make_plots:
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
    controls = generate_control_weights(n=config.controls_n, seed=config.seed)
    all_results["controls"] = controls

    if config.make_plots:
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
    main(parse_args())
