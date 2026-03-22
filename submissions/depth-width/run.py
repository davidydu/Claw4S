"""Run the full depth-vs-width tradeoff experiment suite.

Must be executed from the submissions/depth-width/ directory.
"""

import os
import sys

# Guard: ensure we are in the correct working directory
expected_marker = os.path.join("src", "experiment.py")
if not os.path.isfile(expected_marker):
    print(
        "ERROR: run.py must be executed from submissions/depth-width/\n"
        f"  Current directory: {os.getcwd()}\n"
        f"  Missing expected file: {expected_marker}",
        file=sys.stderr,
    )
    sys.exit(1)

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import run_all_experiments, save_results
from src.report import save_report

print("[1/3] Running depth-vs-width experiments...")
results = run_all_experiments()

print("\n[2/3] Saving results to results/")
save_results(results)

print("\n[3/3] Generating report...")
save_report(results)

print("\nDone. See results/results.json and results/report.md")
