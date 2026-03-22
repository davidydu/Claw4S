"""Run the full activation sparsity analysis and generate all outputs."""

import json
import os
import sys

# Guard: must be run from the submission directory
if not os.path.isfile("requirements.txt"):
    print("ERROR: run.py must be executed from submissions/sparsity/", file=sys.stderr)
    sys.exit(1)

from src.analysis import run_all_experiments
from src.plots import generate_all_plots
from src.report import generate_report

os.makedirs("results", exist_ok=True)

# Run experiments
results = run_all_experiments()

# Generate plots
print("[+] Generating plots...")
plot_paths = generate_all_plots(results, "results")
for p in plot_paths:
    print(f"  Saved: {p}")

# Generate report
print("[+] Generating report...")
report = generate_report(results)
with open("results/report.md", "w") as f:
    f.write(report)
print("  Saved: results/report.md")

# Save raw results (convert to JSON-serializable)
print("[+] Saving results to results/results.json...")
with open("results/results.json", "w") as f:
    json.dump(results, f, indent=2)
print("  Saved: results/results.json")

print("\n[DONE] All results saved to results/")
