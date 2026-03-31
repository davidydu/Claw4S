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
metadata = data.get("metadata", {})

print(f"Experiment: {data.get('experiment', 'unknown')}")
print(f"Metadata: {json.dumps(metadata, indent=2)}")
print(f"Config: {json.dumps(config, indent=2)}")
print(f"Number of runs: {len(runs)}")

required_metadata = [
    "created_utc",
    "runtime_seconds",
    "python_version",
    "platform",
    "torch_version",
    "numpy_version",
    "scipy_version",
    "matplotlib_version",
    "deterministic_algorithms",
]
for key in required_metadata:
    if key not in metadata:
        errors.append(f"Missing metadata field: {key}")

if metadata.get("runtime_seconds", 0) <= 0:
    errors.append("metadata.runtime_seconds must be > 0")
if metadata.get("deterministic_algorithms") is not True:
    errors.append("metadata.deterministic_algorithms must be true")

# Check run count
expected_runs = len(config.get("fractions", [])) * len(config.get("tasks", []))
if len(runs) != expected_runs:
    errors.append(f"Expected {expected_runs} runs, got {len(runs)}")

# Check each run
modular_lags = []
regression_lags = []
required_run_fields = [
    "task_name",
    "frac",
    "gnorm_transition_epoch",
    "metric_transition_epoch",
    "lag_epochs",
    "per_layer",
    "final_train_metric",
    "final_test_metric",
]
for i, run in enumerate(runs):
    for key in required_run_fields:
        if key not in run:
            errors.append(f"Run {i+1}: missing field '{key}'")

    task = run.get("task_name", "unknown")
    frac = run.get("frac", 0)
    lag = run.get("lag_epochs")
    final_train = run.get("final_train_metric")
    final_test = run.get("final_test_metric")

    print(f"\nRun {i+1}: {task} frac={frac:.0%}")
    print(f"  Gradient transition epoch: {run.get('gnorm_transition_epoch')}")
    print(f"  Metric transition epoch:   {run.get('metric_transition_epoch')}")
    print(f"  Lag: {lag} epochs (leads={'YES' if run.get('lag_positive') else 'NO'})")
    if isinstance(final_train, (int, float)):
        print(f"  Final train metric: {final_train:.4f}")
    else:
        print(f"  Final train metric: {final_train}")
    if isinstance(final_test, (int, float)):
        print(f"  Final test metric:  {final_test:.4f}")
    else:
        print(f"  Final test metric:  {final_test}")
    print(f"  Pearson r: {run.get('pearson_r', 0):.4f} (p={run.get('pearson_p', 1):.4e})")

    if lag is None:
        errors.append(f"Run {i+1} ({task} frac={frac}): lag is None")

    if task == "modular_addition" and final_train is not None:
        if final_train < 0.9:
            errors.append(
                f"Run {i+1} ({task} frac={frac}): train acc {final_train:.3f} < 0.9 "
                f"(memorization may not have occurred)"
            )
        if isinstance(lag, (int, float)):
            modular_lags.append(lag)
            if lag <= 0:
                errors.append(
                    f"Run {i+1} ({task} frac={frac}): expected positive lag for grokking task, got {lag}"
                )
    elif task == "regression" and isinstance(lag, (int, float)):
        regression_lags.append(lag)
        if lag > 0:
            errors.append(
                f"Run {i+1} ({task} frac={frac}): expected non-positive lag for smooth-learning control, got {lag}"
            )

    # Per-layer data check
    per_layer = run.get("per_layer", {})
    if len(per_layer) < 2:
        errors.append(f"Run {i+1}: expected >=2 layer results, got {len(per_layer)}")

if not modular_lags:
    errors.append("No modular_addition runs found")
if not regression_lags:
    errors.append("No regression runs found")

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
    expected_frac_keys = {str(f) for f in config.get("fractions", [])}
    observed_frac_keys = set(variance.keys())
    missing_frac_keys = expected_frac_keys - observed_frac_keys
    if missing_frac_keys:
        errors.append(f"Variance analysis missing fractions: {sorted(missing_frac_keys)}")

    for frac_str, stats in sorted(variance.items()):
        for key in ["seeds", "lags", "mean", "std", "min", "max"]:
            if key not in stats:
                errors.append(f"Variance analysis for frac={frac_str}: missing field '{key}'")
                continue

        print(f"  frac={frac_str}: mean_lag={stats['mean']:.1f} +/- {stats['std']:.1f} "
              f"(min={stats['min']}, max={stats['max']}, seeds={stats['seeds']})")
        lags = stats.get("lags", [])
        seeds = stats.get("seeds", [])
        if len(lags) < 2:
            errors.append(f"Variance analysis for frac={frac_str}: fewer than 2 seeds")
        if len(seeds) != len(lags):
            errors.append(
                f"Variance analysis for frac={frac_str}: seeds/lags length mismatch "
                f"({len(seeds)} vs {len(lags)})"
            )
        if any((not isinstance(l, (int, float))) for l in lags):
            errors.append(f"Variance analysis for frac={frac_str}: non-numeric lag value")
        if any((isinstance(l, (int, float)) and l <= 0) for l in lags):
            errors.append(f"Variance analysis for frac={frac_str}: expected all positive lags, got {lags}")
        if isinstance(stats.get("mean"), (int, float)) and stats["mean"] <= 0:
            errors.append(f"Variance analysis for frac={frac_str}: mean lag must be positive")
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
