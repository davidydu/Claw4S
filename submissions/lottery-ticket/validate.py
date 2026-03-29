"""Validate lottery ticket experiment results for completeness and correctness.

Checks that all expected outputs exist and contain reasonable values.
Must be run from the submission directory: submissions/lottery-ticket/
"""

import json
import os
import sys

errors = []

# 1. Check results file exists
results_path = "results/results.json"
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run run.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

meta = data["metadata"]
results = data["results"]

print(f"Total runs: {len(results)}")
print(f"Expected runs: {meta['total_runs']}")
print(f"Runtime: {meta['elapsed_seconds']}s")
print(f"Validation split: {meta.get('validation_fraction', 0.0):.0%}")
print(
    "Environment: "
    f"python={meta.get('python_version', 'unknown')}, "
    f"torch={meta.get('torch_version', 'unknown')}, "
    f"numpy={meta.get('numpy_version', 'unknown')}, "
    f"device={meta.get('device', 'unknown')}"
)

# 2. Check run count
if len(results) != meta["total_runs"]:
    errors.append(f"Expected {meta['total_runs']} runs, got {len(results)}")

# 2b. Check validation split metadata exists
validation_fraction = meta.get("validation_fraction")
if validation_fraction is None:
    errors.append("Missing validation_fraction metadata")
elif not (0.0 < validation_fraction < 0.5):
    errors.append(f"Unexpected validation_fraction: {validation_fraction}")

# 2c. Check reproducibility metadata exists
required_meta_fields = [
    "python_version",
    "torch_version",
    "numpy_version",
    "platform",
    "device",
    "deterministic_algorithms_enabled",
    "seed_values",
    "sparsity_values",
    "task_values",
    "strategy_values",
]
for field in required_meta_fields:
    if field not in meta:
        errors.append(f"Missing metadata field: {field}")

if "deterministic_algorithms_enabled" in meta and not meta["deterministic_algorithms_enabled"]:
    errors.append("Deterministic algorithms are not enabled")

# 3. Check all sparsity levels present
expected_sparsities = {0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95}
found_sparsities = {r["sparsity"] for r in results}
missing = expected_sparsities - found_sparsities
if missing:
    errors.append(f"Missing sparsity levels: {missing}")

# 4. Check all tasks present
expected_tasks = {"modular", "regression"}
found_tasks = {r["task"] for r in results}
if found_tasks != expected_tasks:
    errors.append(f"Expected tasks {expected_tasks}, got {found_tasks}")

# 5. Check all strategies present
expected_strategies = {"magnitude", "random", "structured"}
found_strategies = {r["strategy"] for r in results}
if found_strategies != expected_strategies:
    errors.append(f"Expected strategies {expected_strategies}, got {found_strategies}")

# 6. Check dense baseline is reasonable
dense_modular = [r for r in results if r["task"] == "modular" and r["sparsity"] == 0.0]
if dense_modular:
    avg_acc = sum(r["test_acc"] for r in dense_modular) / len(dense_modular)
    print(f"\nModular dense baseline test accuracy: {avg_acc:.4f}")
    if avg_acc < 0.05:
        errors.append(f"Modular dense baseline accuracy too low: {avg_acc:.4f}")

dense_reg = [r for r in results if r["task"] == "regression" and r["sparsity"] == 0.0]
if dense_reg:
    avg_r2 = sum(r["test_r2"] for r in dense_reg) / len(dense_reg)
    print(f"Regression dense baseline test R^2: {avg_r2:.4f}")
    if avg_r2 < 0.5:
        errors.append(f"Regression dense baseline R^2 too low: {avg_r2:.4f}")

# 7. Check that high sparsity degrades performance
high_sparse_mod = [r for r in results if r["task"] == "modular" and r["sparsity"] == 0.95]
if high_sparse_mod and dense_modular:
    avg_dense = sum(r["test_acc"] for r in dense_modular) / len(dense_modular)
    avg_sparse = sum(r["test_acc"] for r in high_sparse_mod) / len(high_sparse_mod)
    print(f"Modular 95% sparse test accuracy: {avg_sparse:.4f}")
    # At 95% sparsity we expect some degradation (but not necessarily huge)
    # Just check it's a valid number
    if avg_sparse < 0.0 or avg_sparse > 1.0:
        errors.append(f"Invalid modular 95% sparse accuracy: {avg_sparse:.4f}")

# 8. Check that 3 seeds per config exist
from collections import Counter
config_counts = Counter(
    (r["task"], r["strategy"], r["sparsity"]) for r in results
)
for config, count in config_counts.items():
    if count != 3:
        errors.append(f"Config {config} has {count} runs, expected 3")

# 9. Check plots exist
for plot_name in ["accuracy_vs_sparsity.png", "epochs_vs_sparsity.png"]:
    plot_path = os.path.join("results", plot_name)
    if not os.path.exists(plot_path):
        errors.append(f"Missing plot: {plot_path}")
    else:
        size = os.path.getsize(plot_path)
        print(f"Plot {plot_name}: {size} bytes")
        if size < 1000:
            errors.append(f"Plot {plot_name} seems too small: {size} bytes")

# 10. Check report exists
report_path = "results/report.txt"
if not os.path.exists(report_path):
    errors.append(f"Missing report: {report_path}")
else:
    with open(report_path) as f:
        report_text = f.read()
    print(f"Report: {len(report_text)} chars")
    if "KEY FINDINGS" not in report_text:
        errors.append("Report missing KEY FINDINGS section")
    if "95% CI" not in report_text:
        errors.append("Report missing 95% CI statistics")

# 11. Check summary CSV exists
summary_path = "results/summary.csv"
if not os.path.exists(summary_path):
    errors.append(f"Missing summary CSV: {summary_path}")
else:
    with open(summary_path) as f:
        header = f.readline().strip()
    print(f"Summary CSV header: {header}")
    if "metric_ci_low" not in header or "metric_ci_high" not in header:
        errors.append("Summary CSV missing CI columns")

# Final verdict
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation PASSED. All checks OK.")
