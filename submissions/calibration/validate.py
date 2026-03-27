"""Validate calibration experiment results for completeness and correctness.

Usage: .venv/bin/python validate.py
Must be run from submissions/calibration/ directory.
"""

import json
import os
import sys

# Working-directory guard
expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "SKILL.md")
if not os.path.exists(expected_marker):
    print("ERROR: validate.py must be executed from submissions/calibration/",
          file=sys.stderr)
    sys.exit(1)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

results_path = "results/results.json"
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run run.py first.", file=sys.stderr)
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

errors = []
meta = data["metadata"]

# Check metadata
print(f"Hidden widths: {meta['hidden_widths']}")
print(f"Shift magnitudes: {meta['shift_magnitudes']}")
print(f"Seeds: {meta['seeds']}")
print(f"Total experiments: {meta['n_experiments']}")
print(f"Runtime: {meta['elapsed_seconds']}s")

expected_experiments = len(meta['hidden_widths']) * len(meta['seeds'])
actual_experiments = len(data['raw_results'])
print(f"\nRaw results: {actual_experiments} (expected {expected_experiments})")

if actual_experiments != expected_experiments:
    errors.append(f"Expected {expected_experiments} raw results, got {actual_experiments}")

# Check each raw result has all shifts
for r in data['raw_results']:
    n_shifts = len(r['shifts'])
    expected_shifts = len(meta['shift_magnitudes'])
    if n_shifts != expected_shifts:
        errors.append(f"width={r['hidden_width']} seed={r['seed']}: "
                     f"expected {expected_shifts} shifts, got {n_shifts}")

# Check aggregated results
expected_agg = len(meta['hidden_widths']) * len(meta['shift_magnitudes'])
actual_agg = len(data['aggregated'])
print(f"Aggregated results: {actual_agg} (expected {expected_agg})")

if actual_agg != expected_agg:
    errors.append(f"Expected {expected_agg} aggregated entries, got {actual_agg}")

# Validate ECE values are in [0, 1]
for r in data['aggregated']:
    if not (0.0 <= r['ece_mean'] <= 1.0):
        errors.append(f"ECE out of range: {r['ece_mean']} for "
                     f"width={r['hidden_width']}, shift={r['shift_magnitude']}")
    if not (0.0 <= r['accuracy_mean'] <= 1.0):
        errors.append(f"Accuracy out of range: {r['accuracy_mean']} for "
                     f"width={r['hidden_width']}, shift={r['shift_magnitude']}")

# Check ECE generally increases with shift (aggregate across widths)
shifts = sorted(set(r['shift_magnitude'] for r in data['aggregated']))
for width in meta['hidden_widths']:
    eces_by_shift = {}
    for r in data['aggregated']:
        if r['hidden_width'] == width:
            eces_by_shift[r['shift_magnitude']] = r['ece_mean']
    if 0.0 in eces_by_shift and max(shifts) in eces_by_shift:
        id_ece = eces_by_shift[0.0]
        max_ece = eces_by_shift[max(shifts)]
        trend = "increases" if max_ece > id_ece else "DECREASES"
        print(f"  width={width}: ECE {trend} from {id_ece:.4f} (shift=0) "
              f"to {max_ece:.4f} (shift={max(shifts)})")

# Check plot files exist
expected_plots = [
    "results/ece_vs_shift.pdf",
    "results/accuracy_vs_shift.pdf",
    "results/brier_vs_shift.pdf",
    "results/reliability_diagrams.pdf",
    "results/overconfidence_gap.pdf",
]
for plot_path in expected_plots:
    if os.path.exists(plot_path):
        size_kb = os.path.getsize(plot_path) / 1024
        print(f"  Plot: {plot_path} ({size_kb:.1f} KB)")
    else:
        errors.append(f"Missing plot: {plot_path}")

# Check report
report_path = "results/report.md"
if os.path.exists(report_path):
    with open(report_path) as f:
        report = f.read()
    print(f"  Report: {report_path} ({len(report)} chars)")
else:
    errors.append(f"Missing report: {report_path}")

print()
if errors:
    print("VALIDATION FAILED:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation passed.")
