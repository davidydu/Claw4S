#!/usr/bin/env python3
"""Main runner for membership inference scaling experiment.

Must be run from submissions/membership-inference/ directory.
Trains target and shadow models across 5 MLP widths, runs membership
inference attacks, computes correlations, and generates plots + report.
"""

import argparse
import os
import sys
import time
from typing import Any, Dict, List


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


def parse_widths_arg(widths_arg: str) -> List[int]:
    """Parse a comma-separated list of positive integer widths."""
    try:
        widths = [int(part.strip()) for part in widths_arg.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid widths list '{widths_arg}'. Expected comma-separated integers."
        ) from exc

    if not widths:
        raise argparse.ArgumentTypeError("At least one hidden width is required.")

    if any(w <= 0 for w in widths):
        raise argparse.ArgumentTypeError("All hidden widths must be positive integers.")

    return widths


def build_parser(defaults: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build CLI parser with defaults from current experiment settings."""
    parser = argparse.ArgumentParser(
        description=(
            "Run membership inference scaling with configurable model widths, "
            "repeats, shadow models, and synthetic data settings."
        )
    )
    parser.add_argument(
        "--widths",
        type=parse_widths_arg,
        default=defaults["hidden_widths"],
        help=(
            "Comma-separated hidden widths to evaluate "
            f"(default: {','.join(str(w) for w in defaults['hidden_widths'])})."
        ),
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=defaults["n_repeats"],
        help=f"Number of independent repeats per width (default: {defaults['n_repeats']}).",
    )
    parser.add_argument(
        "--n-shadow",
        type=int,
        default=defaults["n_shadow_models"],
        help=f"Number of shadow models per repeat (default: {defaults['n_shadow_models']}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=defaults["seed"],
        help=f"Base random seed (default: {defaults['seed']}).",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=defaults["n_samples"],
        help=f"Synthetic dataset size (default: {defaults['n_samples']}).",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=defaults["n_features"],
        help=f"Number of input features (default: {defaults['n_features']}).",
    )
    parser.add_argument(
        "--n-classes",
        type=int,
        default=defaults["n_classes"],
        help=f"Number of classes (default: {defaults['n_classes']}).",
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=defaults["train_fraction"],
        help=f"Train split fraction in (0,1) (default: {defaults['train_fraction']}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=defaults["output_dir"],
        help=f"Directory for results and plots (default: {defaults['output_dir']}).",
    )
    return parser


def collect_runtime_metadata() -> Dict[str, str]:
    """Capture software versions used for reproducibility metadata."""
    import numpy as np
    import scipy
    import sklearn
    import torch

    metadata: Dict[str, str] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    try:
        import matplotlib

        metadata["matplotlib"] = matplotlib.__version__
    except ImportError:
        metadata["matplotlib"] = "not-installed"
    return metadata


def validate_cli_args(args: argparse.Namespace) -> None:
    """Validate argument ranges and relationships."""
    if args.n_repeats <= 0:
        raise ValueError("--n-repeats must be a positive integer.")
    if args.n_shadow <= 0:
        raise ValueError("--n-shadow must be a positive integer.")
    if args.n_samples <= 0:
        raise ValueError("--n-samples must be a positive integer.")
    if args.n_features <= 0:
        raise ValueError("--n-features must be a positive integer.")
    if args.n_classes <= 1:
        raise ValueError("--n-classes must be greater than 1.")
    if not (0.0 < args.train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1 (exclusive).")
    if not args.output_dir.strip():
        raise ValueError("--output-dir must be a non-empty path.")


def main() -> None:
    check_working_directory()

    start_time = time.time()

    # Import after path check to give clear error if deps are missing
    from src.data import (
        generate_gaussian_clusters,
        split_data,
        SEED,
        N_SAMPLES,
        N_FEATURES,
        N_CLASSES,
    )
    from src.models import HIDDEN_WIDTHS
    from src.attack import run_attack_for_width, N_SHADOW_MODELS
    from src.analysis import (
        compute_correlations,
        generate_plots,
        generate_report,
        save_results,
    )

    defaults: Dict[str, Any] = {
        "hidden_widths": HIDDEN_WIDTHS,
        "n_repeats": 3,
        "n_shadow_models": N_SHADOW_MODELS,
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "n_features": N_FEATURES,
        "n_classes": N_CLASSES,
        "train_fraction": 0.5,
        "output_dir": "results",
    }
    parser = build_parser(defaults)
    args = parser.parse_args()
    validate_cli_args(args)

    output_dir = args.output_dir
    results_path = os.path.join(output_dir, "results.json")
    report_path = os.path.join(output_dir, "report.md")

    experiment_config = {
        "n_samples": args.n_samples,
        "n_features": args.n_features,
        "n_classes": args.n_classes,
        "hidden_widths": args.widths,
        "n_shadow_models": args.n_shadow,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "train_fraction": args.train_fraction,
    }
    runtime = collect_runtime_metadata()

    print("=" * 60)
    print("Membership Inference Scaling Experiment")
    print("=" * 60)
    print(
        "Config: "
        f"widths={args.widths}, repeats={args.n_repeats}, shadows={args.n_shadow}, "
        f"seed={args.seed}"
    )

    # Step 1: Generate target data
    print("\n[1/4] Generating synthetic classification data...")
    X, y = generate_gaussian_clusters(
        n_samples=args.n_samples,
        n_features=args.n_features,
        n_classes=args.n_classes,
        seed=args.seed,
    )
    X_train, y_train, X_test, y_test = split_data(
        X, y, train_fraction=args.train_fraction, seed=args.seed
    )
    print(
        f"  Data: {len(X)} samples, {X.shape[1]} features, "
        f"train={len(X_train)}, test={len(X_test)}"
    )

    # Step 2: Run attacks for each model width
    print("\n[2/4] Running membership inference attacks...")
    all_results = []
    for width in args.widths:
        print(f"  Width={width}...", end=" ", flush=True)
        width_start = time.time()
        result = run_attack_for_width(
            hidden_width=width,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_shadow=args.n_shadow,
            n_repeats=args.n_repeats,
            seed=args.seed,
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

    plots = generate_plots(all_results, output_dir)
    if plots:
        print(f"  Saved {len(plots)} plots to {output_dir}/")
    else:
        print("  (matplotlib not available, skipping plots)")

    # Step 4: Save results
    print(f"\n[4/4] Saving results to {output_dir}/")
    save_results(
        all_results,
        correlations,
        results_path,
        config=experiment_config,
        runtime=runtime,
    )
    generate_report(all_results, correlations, report_path, config=experiment_config)

    total_time = time.time() - start_time
    print(f"\nDone in {total_time:.1f}s")
    print(f"  {results_path}  - raw data")
    print(f"  {report_path}     - summary report")


if __name__ == "__main__":
    try:
        main()
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
