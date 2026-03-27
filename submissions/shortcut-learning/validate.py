"""Validate experiment results for completeness and scientific soundness.

Must be run from the submission directory: submissions/shortcut-learning/
"""

import json
import os
import sys

# Working-directory guard
_expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")
if not os.path.exists(_expected_marker):
    print("ERROR: validate.py must be executed from submissions/shortcut-learning/")
    sys.exit(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

results_path = os.path.join("results", "results.json")
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run run.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

errors = []

# --- Check metadata ---
meta = data.get("metadata", {})
n_configs = meta.get("n_configs", 0)
print(f"Total configurations: {n_configs}")
if n_configs != 45:
    errors.append(f"Expected 45 configurations, got {n_configs}")

# --- Check individual runs ---
runs = data.get("individual_runs", [])
print(f"Individual runs: {len(runs)}")
if len(runs) != 45:
    errors.append(f"Expected 45 individual runs, got {len(runs)}")

# --- Check aggregates ---
aggs = data.get("aggregates", [])
print(f"Aggregate entries: {len(aggs)}")
if len(aggs) != 15:  # 3 hidden_dims x 5 weight_decays
    errors.append(f"Expected 15 aggregate entries, got {len(aggs)}")

# --- Sanity check accuracy ranges ---
for run in runs:
    for key in ["train_acc", "test_acc_with_shortcut", "test_acc_without_shortcut"]:
        val = run.get(key, -1)
        if not (0.0 <= val <= 1.0):
            errors.append(f"Run hd={run['hidden_dim']}, wd={run['weight_decay']}, "
                          f"seed={run['seed']}: {key}={val} out of [0,1]")

# --- Check that shortcut reliance is non-negative for unregularized models ---
no_reg_runs = [r for r in runs if r["weight_decay"] == 0.0]
positive_reliance = sum(1 for r in no_reg_runs if r["shortcut_reliance"] > 0)
print(f"Unregularized runs with positive shortcut reliance: {positive_reliance}/{len(no_reg_runs)}")
if positive_reliance < len(no_reg_runs) // 2:
    errors.append(f"Expected most unregularized runs to show positive shortcut reliance, "
                  f"but only {positive_reliance}/{len(no_reg_runs)} do")

# --- Check that test_acc_with >= test_acc_without on average ---
avg_with = sum(r["test_acc_with_shortcut"] for r in runs) / len(runs)
avg_without = sum(r["test_acc_without_shortcut"] for r in runs) / len(runs)
print(f"Average test acc with shortcut: {avg_with:.4f}")
print(f"Average test acc without shortcut: {avg_without:.4f}")
if avg_with < avg_without:
    errors.append("Average test accuracy with shortcut should be >= without shortcut")

# --- Check findings ---
findings = data.get("findings", [])
print(f"Findings: {len(findings)}")
if len(findings) < 2:
    errors.append(f"Expected >= 2 findings, got {len(findings)}")

# --- Check report file ---
report_path = os.path.join("results", "report.md")
if not os.path.exists(report_path):
    errors.append("results/report.md not found")
else:
    with open(report_path) as f:
        report = f.read()
    if len(report) < 200:
        errors.append(f"Report too short ({len(report)} chars)")
    print(f"Report length: {len(report)} chars")

# --- Verdict ---
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
