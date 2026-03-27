"""Run the full feature attribution consistency experiment.

Must be executed from the submission directory:
    submissions/feature-attribution/
"""

import os
import sys

# Working-directory guard
expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")
if not os.path.exists(expected_marker):
    print("ERROR: run.py must be executed from submissions/feature-attribution/")
    sys.exit(1)

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.experiment import run_experiment

if __name__ == "__main__":
    results = run_experiment()
    print("\nExperiment complete.")
    print(f"Overall mean Spearman rho: {results['summary']['overall_mean_rho']:.4f}")
    print(f"Substantial disagreement: {results['summary']['substantial_disagreement']}")
