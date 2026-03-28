"""Run the full grokking phase diagram analysis.

This script orchestrates:
  1. Phase diagram sweep across weight_decay x dataset_fraction x hidden_dim
  2. Phase classification for each run
  3. Visualization (phase diagram heatmaps + training curves)
  4. Report generation + metadata for reproducibility
"""

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone


def parse_csv_floats(raw: str) -> list[float]:
    """Parse comma-separated floats from CLI."""
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    try:
        return [float(v) for v in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid float list: {raw!r}"
        ) from exc


def parse_csv_ints(raw: str) -> list[int]:
    """Parse comma-separated ints from CLI."""
    values = [chunk.strip() for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    try:
        return [int(v) for v in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid int list: {raw!r}"
        ) from exc


def ensure_submission_directory() -> None:
    """Ensure run.py is invoked from submissions/grokking/."""
    if not os.path.isfile("run.py"):
        raise RuntimeError(
            "run.py must be executed from submissions/grokking/ "
            f"(current directory: {os.getcwd()})"
        )


def set_deterministic_runtime(seed: int) -> None:
    """Pin seeds and deterministic flags for reproducibility."""
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_metadata(
    sweep_config: dict,
    runtime_seconds: float,
    total_runs: int,
    phase_summary: dict,
    torch_version: str,
    numpy_version: str,
) -> dict:
    """Build reproducibility metadata for this run."""
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": round(runtime_seconds, 2),
        "expected_total_runs": total_runs,
        "sweep": sweep_config,
        "phase_counts": phase_summary["phase_counts"],
        "environment": {
            "python_version": platform.python_version(),
            "torch_version": torch_version,
            "numpy_version": numpy_version,
            "platform": platform.platform(),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    from src.sweep import (
        DEFAULT_DATASET_FRACTIONS,
        DEFAULT_EMBED_DIM,
        DEFAULT_HIDDEN_DIMS,
        DEFAULT_MAX_EPOCHS,
        DEFAULT_P,
        DEFAULT_SEED,
        DEFAULT_WEIGHT_DECAYS,
    )

    parser = argparse.ArgumentParser(
        description="Run grokking phase diagram sweep with deterministic outputs."
    )
    parser.add_argument(
        "--weight-decays",
        type=parse_csv_floats,
        default=DEFAULT_WEIGHT_DECAYS,
        help="Comma-separated list (default: 0,0.001,0.01,0.1,1.0).",
    )
    parser.add_argument(
        "--dataset-fractions",
        type=parse_csv_floats,
        default=DEFAULT_DATASET_FRACTIONS,
        help="Comma-separated list (default: 0.3,0.5,0.7,0.9).",
    )
    parser.add_argument(
        "--hidden-dims",
        type=parse_csv_ints,
        default=DEFAULT_HIDDEN_DIMS,
        help="Comma-separated list (default: 16,32,64).",
    )
    parser.add_argument("--p", type=int, default=DEFAULT_P, help="Prime modulus.")
    parser.add_argument(
        "--embed-dim", type=int, default=DEFAULT_EMBED_DIM, help="Embedding dimension."
    )
    parser.add_argument(
        "--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS, help="Max epochs per run."
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED, help="Global reproducibility seed."
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Output directory for generated artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the full analysis pipeline."""
    import numpy as np
    import torch

    from src.analysis import Phase, aggregate_results
    from src.plots import plot_grokking_curves, plot_phase_diagram
    from src.report import generate_report
    from src.sweep import run_sweep

    ensure_submission_directory()
    args = parse_args()
    set_deterministic_runtime(args.seed)

    sweep_config = {
        "weight_decays": args.weight_decays,
        "dataset_fractions": args.dataset_fractions,
        "hidden_dims": args.hidden_dims,
        "p": args.p,
        "embed_dim": args.embed_dim,
        "max_epochs": args.max_epochs,
        "seed": args.seed,
    }
    expected_total_runs = (
        len(args.weight_decays) * len(args.dataset_fractions) * len(args.hidden_dims)
    )

    os.makedirs(args.results_dir, exist_ok=True)

    # Step 1: Run the sweep
    print("[1/4] Running phase diagram sweep...")
    print("=" * 60)
    sweep_start = time.time()
    results = run_sweep(
        weight_decays=args.weight_decays,
        dataset_fractions=args.dataset_fractions,
        hidden_dims=args.hidden_dims,
        p=args.p,
        embed_dim=args.embed_dim,
        max_epochs=args.max_epochs,
        seed=args.seed,
    )
    elapsed = time.time() - sweep_start
    print()

    # Step 2: Save raw results
    print("[2/4] Saving raw results...")
    sweep_path = os.path.join(args.results_dir, "sweep_results.json")
    with open(sweep_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {len(results)} results to {sweep_path}")

    # Save phase diagram summary
    results_for_agg = []
    for r in results:
        phase = r["phase"]
        phase_enum = Phase(phase) if isinstance(phase, str) else phase
        results_for_agg.append(
            {"phase": phase_enum, "grokking_gap": r.get("grokking_gap")}
        )

    stats = aggregate_results(results_for_agg)
    phase_summary = {
        "phase_counts": stats["phase_counts"],
        "total_runs": stats["total_runs"],
        "grokking_fraction": stats["grokking_fraction"],
        "mean_grokking_gap": stats["mean_grokking_gap"],
        "max_grokking_gap": stats["max_grokking_gap"],
    }
    summary_path = os.path.join(args.results_dir, "phase_diagram.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(phase_summary, f, indent=2)
    print(f"  Saved phase summary to {summary_path}")

    metadata = build_metadata(
        sweep_config=sweep_config,
        runtime_seconds=elapsed,
        total_runs=expected_total_runs,
        phase_summary=phase_summary,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
    )
    metadata_path = os.path.join(args.results_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Saved metadata to {metadata_path}")
    print()

    # Step 3: Generate plots
    print("[3/4] Generating plots...")
    hidden_dims = sorted(set(r["config"]["hidden_dim"] for r in results))
    for hd in hidden_dims:
        phase_plot_path = os.path.join(args.results_dir, f"phase_diagram_h{hd}.png")
        plot_phase_diagram(results, hd, phase_plot_path)
    curves_path = os.path.join(args.results_dir, "grokking_curves.png")
    plot_grokking_curves(results, curves_path)
    print()

    # Step 4: Generate report
    print("[4/4] Generating report...")
    report = generate_report(results)
    report_path = os.path.join(args.results_dir, "report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Saved report to {report_path}")
    print()

    # Summary
    print("=" * 60)
    print("Analysis complete!")
    print(f"  Expected runs: {expected_total_runs}")
    print(f"  Actual runs: {stats['total_runs']}")
    for phase in Phase:
        count = stats["phase_counts"].get(phase.value, 0)
        print(f"  {phase.value.capitalize()}: {count}")
    if stats["mean_grokking_gap"] is not None:
        print(f"  Mean grokking gap: {stats['mean_grokking_gap']:.0f} epochs")
    print(f"  Total runtime: {elapsed:.1f}s")
    print()
    print(f"Results saved to {args.results_dir}/")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        sys.exit(1)
