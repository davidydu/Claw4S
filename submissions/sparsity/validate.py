"""Validate sparsity analysis results for completeness and correctness."""

import json
import os
import sys

errors = []

# Check results directory
if not os.path.isdir("results"):
    print("ERROR: results/ directory not found. Run run.py first.")
    sys.exit(1)

# Load results
results_path = "results/results.json"
if not os.path.isfile(results_path):
    print(f"ERROR: {results_path} not found.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

# Check experiments
experiments = data.get("experiments", [])
n_experiments = len(experiments)
print(f"Experiments: {n_experiments} (expected 8)")
if n_experiments != 8:
    errors.append(f"Expected 8 experiments, got {n_experiments}")

# Check each experiment has required fields
required_history_keys = [
    "epochs", "train_loss", "test_loss", "train_acc", "test_acc",
    "dead_neuron_fraction", "near_dead_fraction", "zero_fraction",
    "activation_entropy", "mean_activation_magnitude",
]
for i, exp in enumerate(experiments):
    h = exp.get("history", {})
    for key in required_history_keys:
        if key not in h:
            errors.append(f"Experiment {i}: missing history key '{key}'")

    n_points = len(h.get("epochs", []))
    if n_points < 10:
        errors.append(f"Experiment {i} has only {n_points} tracking points (expected >= 10)")

    # Check dead neuron fraction is in valid range
    dead_fracs = h.get("dead_neuron_fraction", [])
    for df in dead_fracs:
        if not (0.0 <= df <= 1.0):
            errors.append(f"Experiment {i}: dead_neuron_fraction {df} out of [0, 1] range")
            break

    # Check zero fraction is in valid range
    zero_fracs = h.get("zero_fraction", [])
    for zf in zero_fracs:
        if not (0.0 <= zf <= 1.0):
            errors.append(f"Experiment {i}: zero_fraction {zf} out of [0, 1] range")
            break

    # Check test accuracy is non-negative
    test_accs = h.get("test_acc", [])
    for ta in test_accs:
        if ta < -0.01:
            errors.append(f"Experiment {i}: negative test_acc {ta}")
            break

# Check correlations
correlations = data.get("correlations", {})
expected_corrs = [
    "dead_frac_vs_gen_gap", "dead_frac_vs_test_acc",
    "zero_frac_vs_gen_gap", "zero_frac_vs_test_acc",
    "zero_frac_change_vs_test_acc", "sparsity_change_vs_test_acc",
]
for corr_name in expected_corrs:
    if corr_name not in correlations:
        errors.append(f"Missing correlation: {corr_name}")
    else:
        rho = correlations[corr_name].get("rho", None)
        if rho is not None and not (-1.0 <= rho <= 1.0):
            errors.append(f"Correlation {corr_name} rho={rho} out of [-1, 1] range")

print(f"Correlations: {len(correlations)} computed")

# Check experiment summaries
summaries = data.get("experiment_summaries", [])
print(f"Summaries: {len(summaries)}")

# Check grokking analysis
grokking = data.get("grokking_analysis", [])
print(f"Grokking analyses: {len(grokking)}")
if len(grokking) != 4:
    errors.append(f"Expected 4 grokking analyses (one per width), got {len(grokking)}")

# Check hidden widths coverage
widths_seen = set()
for exp in experiments:
    widths_seen.add(exp.get("hidden_dim", 0))
expected_widths = {32, 64, 128, 256}
if widths_seen != expected_widths:
    errors.append(f"Expected widths {expected_widths}, got {widths_seen}")
print(f"Hidden widths: {sorted(widths_seen)}")

# Check tasks
tasks_seen = set()
for exp in experiments:
    tasks_seen.add(exp.get("task", ""))
print(f"Tasks: {sorted(tasks_seen)}")
if len(tasks_seen) < 2:
    errors.append(f"Expected at least 2 tasks, got {len(tasks_seen)}")

# Check generated files
expected_files = ["results/report.md", "results/sparsity_evolution.png",
                  "results/grokking_vs_sparsity.png", "results/width_vs_sparsity.png"]
for fpath in expected_files:
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {fpath}: {size} bytes")
    else:
        errors.append(f"Missing output file: {fpath}")

# Report
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
