"""Run the full Sybil reputation experiment and generate report.

Usage: .venv/bin/python run.py [--diagnostic]
"""

import sys

from src.experiment import run_experiment, run_diagnostic
from src.report import generate_report


def main():
    if "--diagnostic" in sys.argv:
        print("=== Running diagnostic (small grid) ===")
        run_diagnostic()
        print("\nDiagnostic complete.")
        return

    print("=== Running full Sybil reputation experiment ===")
    data = run_experiment()
    report = generate_report(data)
    print("\n" + report)
    print("\n=== Experiment complete ===")


if __name__ == "__main__":
    main()
