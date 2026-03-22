"""Run the full lottery ticket experiment and generate analysis.

Must be run from the submission directory: submissions/lottery-ticket/
"""

import os
import sys

# Guard: ensure we are in the correct working directory
expected_files = ["SKILL.md", "requirements.txt", "run.py"]
for f in expected_files:
    if not os.path.exists(f):
        print(f"ERROR: {f} not found. Run this script from submissions/lottery-ticket/")
        sys.exit(1)

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from src.experiment import run_all_experiments
from src.analysis import (
    compute_summary_stats,
    plot_accuracy_vs_sparsity,
    plot_epochs_vs_sparsity,
    generate_report,
)

print("[1/4] Running experiments...")
results_data = run_all_experiments(output_dir="results")

print("\n[2/4] Computing summary statistics...")
summary = compute_summary_stats(results_data["results"])

print("\n[3/4] Generating plots...")
plot_accuracy_vs_sparsity(summary, output_dir="results")
plot_epochs_vs_sparsity(summary, output_dir="results")

print("\n[4/4] Generating report...")
report = generate_report(results_data, output_dir="results")
print()
print(report)
