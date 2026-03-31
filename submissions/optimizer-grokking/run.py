"""Run the full optimizer grokking landscape experiment.

Must be run from the submission directory (submissions/optimizer-grokking/).
"""

import os
import sys

# Guard: must be run from submission directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: run.py must be executed from submissions/optimizer-grokking/",
          file=sys.stderr)
    sys.exit(1)

# Ensure src/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sweep import run_sweep
from visualize import load_results, plot_heatmap, plot_training_curves, generate_report

print("=" * 60)
print("Optimizer Grokking Landscape Experiment")
print("=" * 60)

# Step 1: Run the sweep
print("\n[1/4] Running optimizer sweep...")
results = run_sweep()

# Step 2: Load and visualize
print("\n[2/4] Generating heatmap...")
data = load_results()
plot_heatmap(data)

# Step 3: Training curves
print("\n[3/4] Generating training curves...")
plot_training_curves(data)

# Step 4: Generate report
print("\n[4/4] Generating summary report...")
report = generate_report(data)

print("\n" + "=" * 60)
print("Experiment complete. Outputs in results/:")
print("  - sweep_results.json   (raw data)")
print("  - grokking_heatmap.png (outcome heatmap)")
print("  - training_curves.png  (representative curves)")
print("  - report.md            (summary report)")
print("=" * 60)
