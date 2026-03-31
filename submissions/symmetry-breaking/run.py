"""Run the full symmetry-breaking experiment suite and generate report.

Must be run from the submissions/symmetry-breaking/ directory.
"""

import os
import sys

# Working-directory guard
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != script_dir:
    print(f"Changing working directory to {script_dir}")
    os.chdir(script_dir)

sys.path.insert(0, script_dir)

from src.trainer import run_all_experiments
from src.analysis import summarize_results, generate_report, save_results
from src.plotting import (
    plot_symmetry_trajectories,
    plot_accuracy_vs_epsilon,
    plot_final_symmetry_heatmap,
)

print("[1/4] Running symmetry-breaking experiments...")
results = run_all_experiments(
    hidden_dims=[16, 32, 64, 128],
    epsilons=[0.0, 1e-6, 1e-4, 1e-2, 1e-1],
    num_epochs=2000,
    batch_size=256,
    lr=0.1,
    log_interval=50,
    seed=42,
)

print("\n[2/4] Computing summary statistics...")
summary = summarize_results(results)

print("\n[3/4] Generating plots...")
plot_symmetry_trajectories(results)
plot_accuracy_vs_epsilon(results)
plot_final_symmetry_heatmap(results)

print("\n[4/4] Saving results to results/")
report = generate_report(results, summary)
save_results(results, summary, report)

print("\nDone. Key files:")
print("  results/results.json       - Raw per-run data")
print("  results/summary.json       - Summary statistics")
print("  results/report.md          - Human-readable report")
print("  results/symmetry_trajectories.png")
print("  results/accuracy_vs_epsilon.png")
print("  results/symmetry_heatmap.png")
