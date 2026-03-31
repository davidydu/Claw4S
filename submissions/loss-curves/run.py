"""Run the full loss curve universality analysis.

Must be run from the submissions/loss-curves/ directory.
"""

import argparse
import os
import sys

# Guard: must be run from the submission directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: run.py must be executed from submissions/loss-curves/")
    sys.exit(1)

from src.analysis import HIDDEN_SIZES, N_EPOCHS, TASKS, run_analysis, save_results
from src.plotting import (
    load_full_curves,
    plot_loss_curves_with_fits,
    plot_aic_comparison,
    plot_exponent_distributions,
    generate_report,
)

def _parse_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(v) for v in _parse_csv(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run loss-curve universality analysis with optional resume/fresh controls."
        )
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=N_EPOCHS,
        help=f"Number of training epochs per run (default: {N_EPOCHS})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for all training runs (default: 42)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=",".join(TASKS),
        help=f"Comma-separated task names (default: {','.join(TASKS)})",
    )
    parser.add_argument(
        "--hidden-sizes",
        type=str,
        default=",".join(str(s) for s in HIDDEN_SIZES),
        help=(
            "Comma-separated hidden sizes "
            f"(default: {','.join(str(s) for s in HIDDEN_SIZES)})"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore results/checkpoint.json and run all configurations fresh.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing results/checkpoint.json before starting.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    tasks = _parse_csv(args.tasks)
    hidden_sizes = _parse_int_csv(args.hidden_sizes)
    unknown_tasks = [task for task in tasks if task not in TASKS]
    if unknown_tasks:
        print(f"ERROR: Unknown task(s): {', '.join(unknown_tasks)}")
        return 1
    if not hidden_sizes:
        print("ERROR: hidden size list cannot be empty")
        return 1

    checkpoint_path = os.path.join("results", "checkpoint.json")
    if args.fresh and os.path.isfile(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"Deleted previous checkpoint: {checkpoint_path}")

    resume = not args.no_resume and not args.fresh

    print("=" * 60)
    print("Loss Curve Universality Analysis")
    print("=" * 60)
    print(
        f"Tasks={tasks}, hidden_sizes={hidden_sizes}, "
        f"epochs={args.epochs}, seed={args.seed}, resume={resume}"
    )

    # Phase 1: Train models and fit curves
    print("\n[1/4] Training models and fitting curves...")
    results = run_analysis(
        tasks=tasks,
        hidden_sizes=hidden_sizes,
        n_epochs=args.epochs,
        seed=args.seed,
        resume=resume,
        checkpoint_path=checkpoint_path,
    )

    # Phase 2: Save results
    print("\n[2/4] Saving results...")
    save_results(results)

    # Phase 3: Generate plots
    print("\n[3/4] Generating plots...")
    curves = load_full_curves()
    plot_loss_curves_with_fits(curves)
    plot_aic_comparison(curves)
    plot_exponent_distributions()

    # Phase 4: Generate and print report
    print("\n[4/4] Generating report...")
    report = generate_report()
    report_path = os.path.join("results", "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"  Saved {report_path}")
    print()
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
