"""Run the full RMT analysis pipeline.

Trains tiny MLPs on modular arithmetic and regression, then analyzes
weight matrix eigenvalue spectra against the Marchenko-Pastur distribution.
"""

import argparse
import json
import os
import platform
import sys

import numpy as np
import torch

# Working-directory guard: must be run from the submission directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: run.py must be executed from the submissions/rmt/ directory.")
    print("  cd submissions/rmt/ && .venv/bin/python run.py")
    sys.exit(1)

from src.data import generate_modular_addition, generate_regression
from src.model import TinyMLP
from src.train import train_model
from src.rmt_analysis import analyze_model_weights
from src.plots import plot_eigenvalue_spectra, plot_ks_summary
from src.report import generate_report, save_report
from src.summary import compute_paired_delta_summary, write_checksum_manifest

DEFAULT_SEED = 42
DEFAULT_HIDDEN_DIMS = [32, 64, 128, 256]
DEFAULT_MOD_EPOCHS = 500
DEFAULT_REG_EPOCHS = 500
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 512
DEFAULT_MODULUS = 97
DEFAULT_REG_SAMPLES = 1000
DEFAULT_OUTPUT_DIR = "results"


def parse_hidden_dims(value: str) -> list[int]:
    """Parse comma-separated hidden dimensions."""
    dims = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        dim = int(token)
        if dim <= 0:
            raise argparse.ArgumentTypeError(
                f"Hidden dims must be positive integers, got {dim}"
            )
        dims.append(dim)
    if not dims:
        raise argparse.ArgumentTypeError(
            "Expected at least one hidden dimension (e.g., 32,64,128)"
        )
    return dims


def build_arg_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Run RMT analysis of trained and untrained tiny MLP weights."
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    parser.add_argument(
        "--hidden-dims",
        type=parse_hidden_dims,
        default=DEFAULT_HIDDEN_DIMS,
        help="Comma-separated hidden dims, e.g. 32,64,128,256.",
    )
    parser.add_argument(
        "--mod-epochs",
        type=int,
        default=DEFAULT_MOD_EPOCHS,
        help="Epochs for modular arithmetic models.",
    )
    parser.add_argument(
        "--reg-epochs",
        type=int,
        default=DEFAULT_REG_EPOCHS,
        help="Epochs for regression models.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LR,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training mini-batch size.",
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=DEFAULT_MODULUS,
        help="Modulus p for modular addition task.",
    )
    parser.add_argument(
        "--reg-samples",
        type=int,
        default=DEFAULT_REG_SAMPLES,
        help="Number of samples for regression task.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated artifacts.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training logs to stage-level progress only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    print("=" * 60)
    print("Random Matrix Theory Analysis of Neural Network Weights")
    print("=" * 60)

    if args.batch_size <= 0:
        print("ERROR: --batch-size must be positive.")
        sys.exit(1)
    if args.mod_epochs <= 0 or args.reg_epochs <= 0:
        print("ERROR: --mod-epochs and --reg-epochs must be positive.")
        sys.exit(1)
    if args.modulus <= 1:
        print("ERROR: --modulus must be > 1.")
        sys.exit(1)
    if args.reg_samples < 10:
        print("ERROR: --reg-samples must be >= 10.")
        sys.exit(1)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    verbose = not args.quiet

    # Step 1: Generate datasets
    print("\n[1/5] Generating datasets...")
    mod_data = generate_modular_addition(p=args.modulus, seed=args.seed)
    reg_data = generate_regression(n_samples=args.reg_samples, seed=args.seed)
    mod_task_label = f"mod{args.modulus}"
    print(f"  Modular addition: {mod_data['n_train']} train, {mod_data['n_test']} test")
    print(f"  Regression: {reg_data['n_train']} train, {reg_data['n_test']} test")

    # Step 2: Train models
    print("\n[2/5] Training models...")
    training_results = []
    trained_models = {}
    untrained_models = {}

    for hidden_dim in args.hidden_dims:
        for task_name, data, epochs in [
            (mod_task_label, mod_data, args.mod_epochs),
            ("regression", reg_data, args.reg_epochs),
        ]:
            model_label = f"{task_name}_h{hidden_dim}"
            print(f"\n  Training {model_label}...")

            # Save untrained model for baseline comparison
            torch.manual_seed(args.seed)
            untrained = TinyMLP(data["input_dim"], hidden_dim, data["output_dim"])
            untrained_models[model_label] = untrained

            # Train a fresh copy
            torch.manual_seed(args.seed)
            model = TinyMLP(data["input_dim"], hidden_dim, data["output_dim"])
            history = train_model(
                model=model,
                X_train=data["X_train"],
                y_train=data["y_train"],
                X_test=data["X_test"],
                y_test=data["y_test"],
                task=data["task"],
                epochs=epochs,
                lr=args.learning_rate,
                seed=args.seed,
                batch_size=args.batch_size,
                verbose=verbose,
            )

            trained_models[model_label] = model
            training_results.append({
                "model_label": model_label,
                "task": task_name,
                "hidden_dim": hidden_dim,
                "epochs": epochs,
                "final_loss": history["final_loss"],
                **{k: v for k, v in history.items()
                   if k in ("final_accuracy", "final_mse")},
            })

    # Step 3: Analyze weight matrices
    print("\n[3/5] Analyzing weight matrices (RMT)...")
    trained_analysis = []
    untrained_analysis = []

    for model_label in sorted(trained_models.keys()):
        print(f"  Analyzing {model_label}...")

        # Trained weights
        trained_weights = [
            (name, W.numpy())
            for name, W in trained_models[model_label].get_weight_matrices()
        ]
        trained_results = analyze_model_weights(trained_weights, model_label)
        trained_analysis.extend(trained_results)

        # Untrained baseline
        untrained_weights = [
            (name, W.numpy())
            for name, W in untrained_models[model_label].get_weight_matrices()
        ]
        untrained_results = analyze_model_weights(
            untrained_weights, model_label
        )
        untrained_analysis.extend(untrained_results)

        # Print summary
        for tr, ut in zip(trained_results, untrained_results):
            print(
                f"    {tr['layer_name']}: "
                f"KS trained={tr['ks_statistic']:.4f} vs "
                f"untrained={ut['ks_statistic']:.4f}, "
                f"outliers={tr['outlier_fraction']:.2f}"
            )

    # Step 4: Generate plots
    print("\n[4/5] Generating plots...")
    os.makedirs(args.output_dir, exist_ok=True)

    spectra_path = plot_eigenvalue_spectra(
        trained_analysis, save_path=os.path.join(args.output_dir, "eigenvalue_spectra.png")
    )
    print(f"  Saved {spectra_path}")

    ks_path = plot_ks_summary(
        trained_analysis, save_path=os.path.join(args.output_dir, "ks_summary.png")
    )
    print(f"  Saved {ks_path}")

    # Also plot untrained for comparison
    untrained_spectra_path = plot_eigenvalue_spectra(
        untrained_analysis,
        save_path=os.path.join(args.output_dir, "eigenvalue_spectra_untrained.png"),
    )
    print(f"  Saved {untrained_spectra_path}")

    # Step 5: Save results and report
    print("\n[5/5] Saving results and report...")

    # Strip eigenvalue arrays from JSON to keep file size reasonable
    def strip_eigenvalues(analysis_list: list[dict]) -> list[dict]:
        return [
            {k: v for k, v in r.items() if k != "eigenvalues"}
            for r in analysis_list
        ]

    delta_ks_summary = compute_paired_delta_summary(
        trained_analysis=trained_analysis,
        untrained_analysis=untrained_analysis,
        seed=args.seed,
    )

    results_data = {
        "metadata": {
            "seed": args.seed,
            "hidden_dims": args.hidden_dims,
            "modulus": args.modulus,
            "regression_samples": args.reg_samples,
            "mod_epochs": args.mod_epochs,
            "reg_epochs": args.reg_epochs,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "output_dir": args.output_dir,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
        },
        "training_results": training_results,
        "trained_analysis": strip_eigenvalues(trained_analysis),
        "untrained_analysis": strip_eigenvalues(untrained_analysis),
        "delta_ks_summary": delta_ks_summary,
    }

    results_path = os.path.join(args.output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2, sort_keys=True)
    print(f"  Saved {results_path}")

    report = generate_report(results_data)
    report_path = save_report(report, os.path.join(args.output_dir, "report.md"))
    print(f"  Saved {report_path}")

    checksums_path = os.path.join(args.output_dir, "checksums.sha256")
    write_checksum_manifest(
        paths=[results_path, report_path, spectra_path, ks_path, untrained_spectra_path],
        manifest_path=checksums_path,
        base_dir=args.output_dir,
    )
    print(f"  Saved {checksums_path}")

    print("\n" + "=" * 60)
    print(f"Analysis complete. See {args.output_dir}/ for outputs.")
    print("=" * 60)


if __name__ == "__main__":
    main()
