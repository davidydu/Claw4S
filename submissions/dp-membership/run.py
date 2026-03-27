"""Run the full membership inference under DP experiment.

Must be executed from the submissions/dp-membership/ directory.
Generates results/results.json and plots in results/.
"""

import json
import os
import sys

# Working directory guard
expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")
if not os.path.exists(expected_marker):
    print("ERROR: run.py must be executed from the submissions/dp-membership/ directory.")
    print(f"  Expected to find: {expected_marker}")
    sys.exit(1)

from src.experiment import run_full_experiment, print_summary
from src.plot import generate_all_plots

# Run experiment
results = run_full_experiment(seeds=[42, 123, 456])

# Save results
os.makedirs("results", exist_ok=True)

# Handle inf values for JSON serialization
def sanitize_for_json(obj):
    """Replace inf/-inf with string representations for JSON."""
    if isinstance(obj, float):
        if obj == float("inf"):
            return "inf"
        if obj == float("-inf"):
            return "-inf"
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    return obj

with open("results/results.json", "w") as f:
    json.dump(sanitize_for_json(results), f, indent=2)
print("\nResults saved to results/results.json")

# Generate plots
plots = generate_all_plots(results, "results")

# Print summary
summary = print_summary(results)
print(summary)

with open("results/summary.txt", "w") as f:
    f.write(summary)
print("\nSummary saved to results/summary.txt")
