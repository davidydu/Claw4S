"""Run the full shortcut learning detection experiment and generate report.

Must be run from the submission directory: submissions/shortcut-learning/
"""

import os
import sys

from src.experiment import run_experiment
from src.report import generate_report


def _ensure_submission_cwd() -> None:
    """Ensure script is run from submissions/shortcut-learning/."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.samefile(os.getcwd(), script_dir):
        print("ERROR: run.py must be executed from submissions/shortcut-learning/")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected: {script_dir}")
        sys.exit(1)


def main() -> None:
    _ensure_submission_cwd()

    results = run_experiment(results_dir="results")
    report = generate_report(results)

    report_path = os.path.join("results", "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nReport written to {report_path}")
    print(report)


if __name__ == "__main__":
    main()
