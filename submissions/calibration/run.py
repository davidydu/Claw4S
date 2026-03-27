"""Run the full calibration-under-shift experiment and generate report.

Usage: .venv/bin/python run.py
Must be run from submissions/calibration/ directory.
"""

import os
import sys
import json

# Working-directory guard
expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "SKILL.md")
if not os.path.exists(expected_marker):
    print("ERROR: run.py must be executed from submissions/calibration/",
          file=sys.stderr)
    sys.exit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.experiment import run_all_experiments
from src.plotting import generate_all_plots
from src.report import generate_report

print("[1/4] Running calibration experiments...")
results = run_all_experiments()

print(f"\n[2/4] Generating plots...")
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)
plot_paths = generate_all_plots(results, output_dir)
print(f"  Generated {len(plot_paths)} plots in {output_dir}/")

print("[3/4] Generating report...")
report = generate_report(results)
report_path = os.path.join(output_dir, "report.md")
with open(report_path, "w") as f:
    f.write(report)
print(f"  Report saved to {report_path}")

print("[4/4] Saving results to results/")
# Save full results (excluding non-serializable objects)
results_path = os.path.join(output_dir, "results.json")
with open(results_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"  Results saved to {results_path}")

print(f"\nDone. {results['metadata']['n_experiments']} experiments completed "
      f"in {results['metadata']['elapsed_seconds']:.1f}s.")
print(report)
