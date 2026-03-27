"""Validate experiment results for completeness and correctness.

Must be executed from the submission directory:
    submissions/feature-attribution/
"""

import json
import os
import sys

# Working-directory guard
expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SKILL.md")
if not os.path.exists(expected_marker):
    print("ERROR: validate.py must be executed from submissions/feature-attribution/")
    sys.exit(1)

results_path = os.path.join("results", "results.json")
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run run.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

errors = []

# Check metadata
meta = data.get("metadata", {})
depths = meta.get("depths", [])
seeds = meta.get("seeds", [])

if len(depths) < 3:
    errors.append(f"Expected >= 3 depths, got {len(depths)}")
if len(seeds) < 3:
    errors.append(f"Expected >= 3 seeds, got {len(seeds)}")

print(f"Depths: {depths}")
print(f"Seeds: {seeds}")
print(f"Samples: {meta.get('n_samples', '?')}, Test: {meta.get('n_test', '?')}")
print(f"Features: {meta.get('n_features', '?')}, Classes: {meta.get('n_classes', '?')}")
print(f"Runtime: {meta.get('elapsed_seconds', '?')}s")

# Check per-depth results
per_depth = data.get("per_depth", {})
if len(per_depth) != len(depths):
    errors.append(f"Expected {len(depths)} depth entries, got {len(per_depth)}")

expected_pairs = [
    "vanilla_gradient_vs_gradient_x_input",
    "vanilla_gradient_vs_integrated_gradients",
    "gradient_x_input_vs_integrated_gradients",
]

for d_str, d_data in per_depth.items():
    acc = d_data.get("accuracy_mean", 0)
    print(f"\nDepth {d_str}:")
    print(f"  Accuracy: {acc:.3f} +/- {d_data.get('accuracy_std', 0):.3f}")

    if acc < 0.5:
        errors.append(f"Depth {d_str}: accuracy {acc:.3f} < 0.5 (training may have failed)")

    agreement = d_data.get("agreement", {})
    for pair_key in expected_pairs:
        if pair_key not in agreement:
            errors.append(f"Depth {d_str}: missing agreement pair {pair_key}")
        else:
            s = agreement[pair_key]
            rho = s["mean"]
            std = s["std"]
            print(f"  {pair_key}: rho={rho:.3f} +/- {std:.3f}")

            if abs(rho) > 1.0:
                errors.append(f"Depth {d_str}, {pair_key}: rho={rho:.3f} out of [-1, 1]")

    per_seed = d_data.get("per_seed", [])
    if len(per_seed) != len(seeds):
        errors.append(f"Depth {d_str}: expected {len(seeds)} seed entries, got {len(per_seed)}")

# Check summary
summary = data.get("summary", {})
overall_rho = summary.get("overall_mean_rho", None)
if overall_rho is None:
    errors.append("Missing summary.overall_mean_rho")
else:
    print(f"\nOverall mean Spearman rho: {overall_rho:.4f}")
    print(f"Substantial disagreement: {summary.get('substantial_disagreement', '?')}")

# Check report file
report_path = os.path.join("results", "report.md")
if not os.path.exists(report_path):
    errors.append("Missing results/report.md")
else:
    with open(report_path) as f:
        report_len = len(f.read())
    print(f"Report: {report_len} characters")

# Final verdict
print(f"\n{'='*40}")
if errors:
    print(f"VALIDATION FAILED: {len(errors)} error(s)")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("VALIDATION PASSED: All checks OK")
    sys.exit(0)
