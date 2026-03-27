"""Run the full shortcut learning detection experiment and generate report.

Must be run from the submission directory: submissions/shortcut-learning/
"""

import os
import sys

# Working-directory guard: ensure we are in the submission directory
_expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")
if not os.path.exists(_expected_marker):
    print("ERROR: run.py must be executed from submissions/shortcut-learning/")
    sys.exit(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.experiment import run_experiment
from src.report import generate_report

results = run_experiment(results_dir="results")
report = generate_report(results)

report_path = os.path.join("results", "report.md")
with open(report_path, "w") as f:
    f.write(report)
print(f"\nReport written to {report_path}")
print(report)
