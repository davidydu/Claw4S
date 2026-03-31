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
config = data.get("config", {})

tasks_seen = {exp.get("task", "") for exp in experiments if exp.get("task")}
widths_seen = {exp.get("hidden_dim", 0) for exp in experiments}

configured_widths = set(config.get("hidden_widths", []))
if configured_widths:
    expected_widths = configured_widths
else:
    expected_widths = {32, 64, 128, 256}

expected_experiments = len(expected_widths) * max(1, len(tasks_seen))
print(f"Experiments: {n_experiments} (expected {expected_experiments})")
if n_experiments != expected_experiments:
    errors.append(f"Expected {expected_experiments} experiments, got {n_experiments}")

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
required_corr_fields = ["rho", "p_value", "n", "ci_low", "ci_high"]
expected_corrs = [
    "dead_frac_vs_gen_gap", "dead_frac_vs_test_acc",
    "zero_frac_vs_gen_gap", "zero_frac_vs_test_acc",
    "zero_frac_change_vs_test_acc", "sparsity_change_vs_test_acc",
]
for corr_name in expected_corrs:
    if corr_name not in correlations:
        errors.append(f"Missing correlation: {corr_name}")
    else:
        corr_vals = correlations[corr_name]
        for field in required_corr_fields:
            if field not in corr_vals:
                errors.append(f"Correlation {corr_name} missing field '{field}'")
        rho = corr_vals.get("rho", None)
        if rho is not None and not (-1.0 <= rho <= 1.0):
            errors.append(f"Correlation {corr_name} rho={rho} out of [-1, 1] range")
        n_samples = corr_vals.get("n", None)
        if n_samples is not None and n_samples <= 0:
            errors.append(f"Correlation {corr_name} has invalid n={n_samples}")

print(f"Correlations: {len(correlations)} computed")

# Check task-stratified correlations
correlations_by_task = data.get("correlations_by_task", {})
print(f"Task-stratified correlation groups: {len(correlations_by_task)}")
if not correlations_by_task:
    errors.append("Missing correlations_by_task")
else:
    for task in tasks_seen:
        if task not in correlations_by_task:
            errors.append(f"Missing task-stratified correlations for task '{task}'")
            continue
        task_corrs = correlations_by_task[task]
        for corr_name in expected_corrs:
            if corr_name not in task_corrs:
                errors.append(
                    f"Task '{task}' missing correlation '{corr_name}'"
                )
                continue
            corr_vals = task_corrs[corr_name]
            for field in required_corr_fields:
                if field not in corr_vals:
                    errors.append(
                        f"Task '{task}' correlation {corr_name} missing field '{field}'"
                    )

# Check experiment summaries
summaries = data.get("experiment_summaries", [])
print(f"Summaries: {len(summaries)}")

# Check grokking analysis
grokking = data.get("grokking_analysis", [])
print(f"Grokking analyses: {len(grokking)}")
expected_grokking = sum(1 for exp in experiments if "modular_addition" in exp.get("task", ""))
if len(grokking) != expected_grokking:
    errors.append(
        f"Expected {expected_grokking} grokking analyses (one per modular-addition run), got {len(grokking)}"
    )

# Check hidden widths coverage
if widths_seen != expected_widths:
    errors.append(f"Expected widths {expected_widths}, got {widths_seen}")
print(f"Hidden widths: {sorted(widths_seen)}")

# Check tasks
print(f"Tasks: {sorted(tasks_seen)}")
if len(tasks_seen) < 2:
    errors.append(f"Expected at least 2 tasks, got {len(tasks_seen)}")

# Check config metadata
required_config_keys = [
    "hidden_widths", "n_epochs", "track_every",
    "mod_add_lr", "mod_add_wd", "reg_lr", "reg_wd", "seed",
]
for key in required_config_keys:
    if key not in config:
        errors.append(f"Missing config key: {key}")

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
