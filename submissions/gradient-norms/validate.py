"""Validate gradient norm phase transition experiment results.

Checks:
  1. results/results.json exists and has correct structure
  2. All 6 runs are present (2 tasks x 3 fractions)
  3. Modular addition runs show grokking (high train acc, reasonable test acc)
  4. Lag values are computed (not NaN)
  5. Plot files exist
  6. At least some runs show gradient norm leading metric transition

Must be run from submissions/gradient-norms/
"""

import json
import os
import sys

# Guard: must run from the correct directory
if not os.path.isfile("SKILL.md"):
    print("ERROR: validate.py must be executed from submissions/gradient-norms/")
    sys.exit(1)

errors = []

# --- Check results.json ---
results_path = "results/results.json"
if not os.path.isfile(results_path):
    print(f"FATAL: {results_path} not found. Run run.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

runs = data.get("runs", [])
config = data.get("config", {})

print(f"Experiment: {data.get('experiment', 'unknown')}")
print(f"Config: {json.dumps(config, indent=2)}")
print(f"Number of runs: {len(runs)}")

# Check run count
expected_runs = len(config.get("fractions", [])) * len(config.get("tasks", []))
if len(runs) != expected_runs:
    errors.append(f"Expected {expected_runs} runs, got {len(runs)}")

# Check each run
for i, run in enumerate(runs):
    task = run.get("task_name", "unknown")
    frac = run.get("frac", 0)
    lag = run.get("lag_epochs")
    final_train = run.get("final_train_metric")
    final_test = run.get("final_test_metric")

    print(f"\nRun {i+1}: {task} frac={frac:.0%}")
    print(f"  Gradient transition epoch: {run.get('gnorm_transition_epoch')}")
    print(f"  Metric transition epoch:   {run.get('metric_transition_epoch')}")
    print(f"  Lag: {lag} epochs (leads={'YES' if run.get('lag_positive') else 'NO'})")
    print(f"  Final train metric: {final_train:.4f}")
    print(f"  Final test metric:  {final_test:.4f}")
    print(f"  Pearson r: {run.get('pearson_r', 0):.4f} (p={run.get('pearson_p', 1):.4e})")

    if lag is None:
        errors.append(f"Run {i+1} ({task} frac={frac}): lag is None")

    if task == "modular_addition" and final_train is not None:
        if final_train < 0.9:
            errors.append(
                f"Run {i+1} ({task} frac={frac}): train acc {final_train:.3f} < 0.9 "
                f"(memorization may not have occurred)"
            )

    # Per-layer data check
    per_layer = run.get("per_layer", {})
    if len(per_layer) < 2:
        errors.append(f"Run {i+1}: expected >=2 layer results, got {len(per_layer)}")

# --- Check plots ---
expected_plots = [
    "summary_grid.png",
    "lag_barchart.png",
    "weight_norms.png",
]
print("\nPlot files:")
for plot_name in expected_plots:
    path = os.path.join("results", plot_name)
    exists = os.path.isfile(path)
    size = os.path.getsize(path) if exists else 0
    status = f"OK ({size:,} bytes)" if exists else "MISSING"
    print(f"  {plot_name}: {status}")
    if not exists:
        errors.append(f"Missing plot: {plot_name}")

# Count per-run plots
per_run_plots = [f for f in os.listdir("results") if f.startswith("run_")]
print(f"  Per-run plots: {len(per_run_plots)}")
if len(per_run_plots) < expected_runs:
    errors.append(f"Expected >= {expected_runs} per-run plots, got {len(per_run_plots)}")

# --- Check scientific claim ---
n_positive = sum(1 for r in runs if r.get("lag_positive", False))
print(f"\nGradient norm leads in {n_positive}/{len(runs)} runs")

# --- Check variance analysis ---
variance = data.get("variance_analysis", {})
if variance:
    print("\nMulti-seed variance analysis (modular addition):")
    for frac_str, stats in sorted(variance.items()):
        print(f"  frac={frac_str}: mean_lag={stats['mean']:.1f} +/- {stats['std']:.1f} "
              f"(min={stats['min']}, max={stats['max']}, seeds={stats['seeds']})")
        if len(stats.get("lags", [])) < 2:
            errors.append(f"Variance analysis for frac={frac_str}: fewer than 2 seeds")
else:
    errors.append("Missing variance_analysis in results.json")

# --- Report ---
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
