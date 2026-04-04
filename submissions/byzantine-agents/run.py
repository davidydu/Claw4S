"""Run the full Byzantine fault tolerance experiment and generate report."""

import sys

from src.experiment import run_experiment, save_results
from src.output_spec import clear_primary_artifacts
from src.report import generate_report, save_report


def main() -> int:
    removed = clear_primary_artifacts()
    if removed:
        print(f"[0/3] Removed {len(removed)} stale artifact(s).")

    print("[1/3] Running experiment (405 configs with multiprocessing)...")
    results = run_experiment()

    print(f"[2/3] Saving results to results/results.json ({results['metadata']['elapsed_seconds']:.1f}s elapsed)")
    save_results(results)

    print("[3/3] Generating report...")
    report = generate_report(results)
    save_report(report)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
