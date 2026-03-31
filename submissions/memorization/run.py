"""Run the memorization capacity scaling experiment."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import torch

# Ensure we are running from the submission directory.
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from src.analysis import analyze_results
from src.plots import plot_memorization_curves, plot_threshold_comparison
from src.report import generate_report
from src.sweep import (
    DEFAULT_D,
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_LR,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_N_CLASSES,
    DEFAULT_N_TEST,
    DEFAULT_N_TRAIN,
    run_multi_seed_sweep,
)


def parse_int_list(value: str) -> list[int]:
    """Parse a comma-separated list of positive integers."""
    items = [x.strip() for x in value.split(",") if x.strip()]
    if not items:
        raise argparse.ArgumentTypeError("list must contain at least one integer")
    parsed: list[int] = []
    for item in items:
        try:
            number = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                f"invalid integer '{item}' in list '{value}'"
            ) from exc
        if number <= 0:
            raise argparse.ArgumentTypeError(
                f"all values must be positive, got {number}"
            )
        parsed.append(number)
    return parsed


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for configurable experiment runs."""
    parser = argparse.ArgumentParser(
        description="Run memorization-capacity sweep with configurable settings."
    )
    parser.add_argument(
        "--seeds",
        type=parse_int_list,
        default=[42, 43, 44],
        help="Comma-separated random seeds (default: 42,43,44).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_int_list,
        default=DEFAULT_HIDDEN_DIMS,
        help=(
            "Comma-separated hidden widths to sweep "
            f"(default: {','.join(map(str, DEFAULT_HIDDEN_DIMS))})."
        ),
    )
    parser.add_argument("--n-train", type=int, default=DEFAULT_N_TRAIN)
    parser.add_argument("--n-test", type=int, default=DEFAULT_N_TEST)
    parser.add_argument("--d", type=int, default=DEFAULT_D)
    parser.add_argument("--n-classes", type=int, default=DEFAULT_N_CLASSES)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Output directory for report/results/figures (default: results).",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation (faster smoke runs).",
    )
    parser.add_argument(
        "--no-clean-output",
        action="store_true",
        help="Do not delete existing output directory before running.",
    )
    return parser


def clean_output_dir(output_dir: Path, clean_enabled: bool) -> None:
    """Delete output directory before run when clean mode is enabled."""
    if not clean_enabled or not output_dir.exists():
        return
    if output_dir in (Path("/"), SCRIPT_DIR):
        raise ValueError(f"Refusing to delete unsafe output dir: {output_dir}")
    shutil.rmtree(output_dir)
    print(f"  Cleared existing output directory: {output_dir}")


def build_run_metadata(
    args: argparse.Namespace, start_utc: str, end_utc: str
) -> dict:
    """Build reproducibility metadata for results.json."""
    return {
        "seeds": args.seeds,
        "hidden_dims": args.hidden_dims,
        "n_train": args.n_train,
        "n_test": args.n_test,
        "d": args.d,
        "n_classes": args.n_classes,
        "max_epochs": args.max_epochs,
        "lr": args.lr,
        "output_dir": args.output_dir,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "dependency_versions": {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "execution": {
            "start_utc": start_utc,
            "end_utc": end_utc,
            "plots_generated": not args.no_plots,
        },
    }


def augment_results_json(results_path: Path, run_metadata: dict) -> None:
    """Inject run metadata into results JSON for downstream validation."""
    with results_path.open() as f:
        payload = json.load(f)
    payload["run_metadata"] = run_metadata
    with results_path.open("w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved run metadata to {results_path}")


def run_experiment(args: argparse.Namespace) -> int:
    """Execute full experiment pipeline and return process exit code."""
    if min(args.n_train, args.n_test, args.d, args.n_classes, args.max_epochs) <= 0:
        raise ValueError("n_train, n_test, d, n_classes, and max_epochs must be > 0")
    if args.lr <= 0:
        raise ValueError("lr must be > 0")

    output_dir = Path(args.output_dir)
    clean_output_dir(output_dir=output_dir, clean_enabled=not args.no_clean_output)

    print("=" * 60)
    print("Memorization Capacity Scaling Experiment")
    print("=" * 60)
    print("Configuration:")
    print(f"  seeds={args.seeds}")
    print(f"  hidden_dims={args.hidden_dims}")
    print(
        f"  n_train={args.n_train}, n_test={args.n_test}, d={args.d}, n_classes={args.n_classes}"
    )
    print(f"  max_epochs={args.max_epochs}, lr={args.lr}, output_dir={output_dir}")

    start_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Step 1: Run multi-seed sweep.
    print(f"\n[1/4] Running model size sweep ({len(args.seeds)} seed(s))...")
    multi_results = run_multi_seed_sweep(
        seeds=args.seeds,
        hidden_dims=args.hidden_dims,
        n_train=args.n_train,
        n_test=args.n_test,
        d=args.d,
        n_classes=args.n_classes,
        max_epochs=args.max_epochs,
        lr=args.lr,
    )

    # Use first seed as the primary sweep for downstream analysis.
    primary_sweep = {
        "metadata": {
            k: v
            for k, v in multi_results["metadata"].items()
            if k not in ("seeds", "n_seeds")
        },
        "results": multi_results["per_seed_results"][0],
    }
    primary_sweep["metadata"]["seed"] = multi_results["metadata"]["seeds"][0]

    # Step 2: Analyze results.
    print("\n[2/4] Analyzing results...")
    analysis = analyze_results(primary_sweep)
    analysis["multi_seed"] = {
        "seeds": multi_results["metadata"]["seeds"],
        "aggregated": multi_results["aggregated"],
    }

    # Step 3: Generate plots.
    print("\n[3/4] Generating plots...")
    if args.no_plots:
        print("  Skipping plots (--no-plots).")
    else:
        figures_dir = output_dir / "figures"
        plot_memorization_curves(analysis, output_dir=str(figures_dir))
        plot_threshold_comparison(analysis, output_dir=str(figures_dir))

    # Step 4: Generate report and JSON results.
    print("\n[4/4] Generating report...")
    generate_report(primary_sweep, analysis, output_dir=str(output_dir))
    end_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    run_metadata = build_run_metadata(args=args, start_utc=start_utc, end_utc=end_utc)
    augment_results_json(output_dir / "results.json", run_metadata=run_metadata)

    print("\n" + "=" * 60)
    print("Experiment complete. Key results:")
    for label_type in ["random", "structured"]:
        lt = analysis["label_types"][label_type]
        threshold = lt["threshold"]
        sig = lt["sigmoid_fit"]
        print(f"  {label_type} labels:")
        print(f"    Max train acc: {lt['max_train_acc']:.4f}")
        if threshold["achieved"]:
            print(f"    Interpolation threshold: {threshold['threshold_params']:,} params")
        if sig["fit_success"]:
            print(
                f"    Sigmoid sharpness: {sig['sharpness']:.2f} "
                f"(R^2={sig['r_squared']:.4f})"
            )

    print(f"\nMulti-seed variance ({len(args.seeds)} seed(s)):")
    for entry in multi_results["aggregated"]:
        if entry["hidden_dim"] in [5, 20, 80, 640]:
            print(
                f"  {entry['label_type']:>10s} h={entry['hidden_dim']:>3d}: "
                f"train_acc={entry['train_acc_mean']:.4f} +/- {entry['train_acc_std']:.4f}, "
                f"test_acc={entry['test_acc_mean']:.4f} +/- {entry['test_acc_std']:.4f}"
            )
    print("=" * 60)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse args and run experiment."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run_experiment(args)


if __name__ == "__main__":
    raise SystemExit(main())
