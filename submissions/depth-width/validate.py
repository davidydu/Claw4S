"""Validate depth-vs-width experiment results for completeness and correctness.

Must be executed from the submissions/depth-width/ directory.
"""

import json
import os
import sys

# Guard: ensure we are in the correct working directory
if not os.path.isfile(os.path.join("src", "experiment.py")):
    print(
        "ERROR: validate.py must be executed from submissions/depth-width/",
        file=sys.stderr,
    )
    sys.exit(1)

results_path = os.path.join("results", "results.json")
report_path = os.path.join("results", "report.md")

errors = []

# Check files exist
if not os.path.isfile(results_path):
    errors.append(f"Missing results file: {results_path}")
if not os.path.isfile(report_path):
    errors.append(f"Missing report file: {report_path}")

if errors:
    for e in errors:
        print(f"FAIL: {e}")
    sys.exit(1)

# Load and validate results
with open(results_path) as f:
    data = json.load(f)

metadata = data.get("metadata", {})
results = data.get("results", [])

print(f"Experiments run: {len(results)}")
print(f"Parameter budgets: {metadata.get('param_budgets')}")
print(f"Depths tested: {metadata.get('depths')}")
print(f"Seed: {metadata.get('seed')}")
print(f"PyTorch version: {metadata.get('torch_version')}")

# Check expected experiment count
expected_budgets = metadata.get("param_budgets", [])
expected_depths = metadata.get("depths", [])
num_tasks = 2  # sparse_parity and smooth_regression
expected_count = len(expected_budgets) * len(expected_depths) * num_tasks

if len(results) != expected_count:
    errors.append(
        f"Expected {expected_count} experiments, got {len(results)}"
    )

# Check each result has required fields
required_fields = ["param_budget", "num_hidden_layers", "task_name"]
for i, r in enumerate(results):
    if r.get("skipped"):
        continue
    for field in required_fields:
        if field not in r:
            errors.append(f"Experiment {i} missing field: {field}")

# Check non-skipped results have metrics
completed = [r for r in results if not r.get("skipped")]
print(f"Completed experiments: {len(completed)} / {len(results)}")

for r in completed:
    if "best_test_metric" not in r:
        errors.append(
            f"Missing best_test_metric for {r.get('task_name')} "
            f"depth={r.get('num_hidden_layers')} "
            f"budget={r.get('param_budget')}"
        )

# Check both tasks are present
task_names = set(r["task_name"] for r in results)
if "sparse_parity" not in task_names:
    errors.append("Missing sparse_parity task results")
if "smooth_regression" not in task_names:
    errors.append("Missing smooth_regression task results")

# Check that param counts are within 20% of budget
for r in completed:
    budget = r["param_budget"]
    actual = r.get("actual_params", 0)
    if actual == 0:
        errors.append(
            f"Missing actual_params for depth={r['num_hidden_layers']} "
            f"budget={budget}"
        )
    elif abs(actual - budget) / budget > 0.20:
        errors.append(
            f"Param count {actual} deviates >20% from budget {budget} "
            f"(depth={r['num_hidden_layers']})"
        )

# Summary by task
for task in sorted(task_names):
    task_results = [r for r in completed if r["task_name"] == task]
    if task_results:
        metric_name = task_results[0].get("metric_name", "metric")
        best = max(task_results, key=lambda r: r["best_test_metric"])
        worst = min(task_results, key=lambda r: r["best_test_metric"])
        print(
            f"\n  {task} ({metric_name}):"
            f"\n    Best:  depth={best['num_hidden_layers']}, "
            f"budget={best['param_budget']//1000}K, "
            f"{metric_name}={best['best_test_metric']:.4f}"
            f"\n    Worst: depth={worst['num_hidden_layers']}, "
            f"budget={worst['param_budget']//1000}K, "
            f"{metric_name}={worst['best_test_metric']:.4f}"
        )

# Check report is non-empty
report_size = os.path.getsize(report_path)
if report_size < 100:
    errors.append(f"Report file too small ({report_size} bytes)")

print(f"\nReport size: {report_size} bytes")

# Final verdict
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
