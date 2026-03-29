"""Validate calibration experiment results for completeness and correctness.

Usage: .venv/bin/python validate.py
Must be run from submissions/calibration/ directory.
"""

import json
import math
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

repro = meta.get('reproducibility')
required_repro_keys = [
    'python_version',
    'python_implementation',
    'torch_version',
    'numpy_version',
    'torch_deterministic_algorithms_enabled',
    'torch_num_threads',
]
if not isinstance(repro, dict):
    errors.append("Missing metadata.reproducibility dict")
else:
    missing_repro = [k for k in required_repro_keys if k not in repro]
    if missing_repro:
        errors.append("Missing reproducibility keys: " + ", ".join(missing_repro))
    else:
        print(f"Python: {repro['python_implementation']} {repro['python_version']}")
        print(f"Torch: {repro['torch_version']}, NumPy: {repro['numpy_version']}")
        print(
            "Deterministic algorithms enabled: "
            f"{repro['torch_deterministic_algorithms_enabled']}"
        )
        print(f"Torch threads: {repro['torch_num_threads']}")
        if repro['torch_deterministic_algorithms_enabled'] is not True:
            errors.append("torch_deterministic_algorithms_enabled must be True")
        if (
            not isinstance(repro['torch_num_threads'], int)
            or repro['torch_num_threads'] < 1
        ):
            errors.append("torch_num_threads must be an integer >= 1")

expected_experiments = len(meta['hidden_widths']) * len(meta['seeds'])
actual_experiments = len(data['raw_results'])
print(f"\nRaw results: {actual_experiments} (expected {expected_experiments})")

if actual_experiments != expected_experiments:
    errors.append(f"Expected {expected_experiments} raw results, got {actual_experiments}")

# Check each raw result has all shifts
expected_shift_keys = {str(mag) for mag in meta['shift_magnitudes']}
for r in data['raw_results']:
    n_shifts = len(r['shifts'])
    expected_shifts = len(meta['shift_magnitudes'])
    if n_shifts != expected_shifts:
        errors.append(f"width={r['hidden_width']} seed={r['seed']}: "
                     f"expected {expected_shifts} shifts, got {n_shifts}")
    if set(r['shifts'].keys()) != expected_shift_keys:
        errors.append(
            f"width={r['hidden_width']} seed={r['seed']}: "
            "shift keys do not match metadata shift magnitudes"
        )
    if not math.isfinite(r['final_train_loss']):
        errors.append(
            f"width={r['hidden_width']} seed={r['seed']}: "
            f"final_train_loss is not finite ({r['final_train_loss']})"
        )

    for shift_key, shift_data in r['shifts'].items():
        if not (0.0 <= shift_data['ece'] <= 1.0):
            errors.append(
                f"ECE out of range in raw results: {shift_data['ece']} "
                f"for width={r['hidden_width']}, seed={r['seed']}, shift={shift_key}"
            )
        if not (0.0 <= shift_data['accuracy'] <= 1.0):
            errors.append(
                f"Accuracy out of range in raw results: {shift_data['accuracy']} "
                f"for width={r['hidden_width']}, seed={r['seed']}, shift={shift_key}"
            )
        if not (0.0 <= shift_data['brier_score'] <= 2.0):
            errors.append(
                f"Brier score out of range in raw results: {shift_data['brier_score']} "
                f"for width={r['hidden_width']}, seed={r['seed']}, shift={shift_key}"
            )
        if not (0.0 <= shift_data['mean_confidence'] <= 1.0):
            errors.append(
                f"Mean confidence out of range in raw results: "
                f"{shift_data['mean_confidence']} for width={r['hidden_width']}, "
                f"seed={r['seed']}, shift={shift_key}"
            )
        reliability = shift_data.get('reliability', {})
        if len(reliability.get('bin_accs', [])) != meta['n_bins']:
            errors.append(
                f"Invalid reliability bin_accs length for width={r['hidden_width']}, "
                f"seed={r['seed']}, shift={shift_key}"
            )
        if len(reliability.get('bin_confs', [])) != meta['n_bins']:
            errors.append(
                f"Invalid reliability bin_confs length for width={r['hidden_width']}, "
                f"seed={r['seed']}, shift={shift_key}"
            )
        if len(reliability.get('bin_counts', [])) != meta['n_bins']:
            errors.append(
                f"Invalid reliability bin_counts length for width={r['hidden_width']}, "
                f"seed={r['seed']}, shift={shift_key}"
            )
        if len(reliability.get('bin_edges', [])) != meta['n_bins'] + 1:
            errors.append(
                f"Invalid reliability bin_edges length for width={r['hidden_width']}, "
                f"seed={r['seed']}, shift={shift_key}"
            )

# Check aggregated results
expected_agg = len(meta['hidden_widths']) * len(meta['shift_magnitudes'])
actual_agg = len(data['aggregated'])
print(f"Aggregated results: {actual_agg} (expected {expected_agg})")

if actual_agg != expected_agg:
    errors.append(f"Expected {expected_agg} aggregated entries, got {actual_agg}")
else:
    expected_pairs = {
        (width, float(shift))
        for width in meta['hidden_widths']
        for shift in meta['shift_magnitudes']
    }
    actual_pairs = {
        (r['hidden_width'], float(r['shift_magnitude']))
        for r in data['aggregated']
    }
    missing_pairs = expected_pairs - actual_pairs
    extra_pairs = actual_pairs - expected_pairs
    if missing_pairs:
        errors.append(f"Missing aggregated pairs: {sorted(missing_pairs)}")
    if extra_pairs:
        errors.append(f"Unexpected aggregated pairs: {sorted(extra_pairs)}")

# Validate ECE values are in [0, 1]
for r in data['aggregated']:
    if not (0.0 <= r['ece_mean'] <= 1.0):
        errors.append(f"ECE out of range: {r['ece_mean']} for "
                     f"width={r['hidden_width']}, shift={r['shift_magnitude']}")
    if not (0.0 <= r['accuracy_mean'] <= 1.0):
        errors.append(f"Accuracy out of range: {r['accuracy_mean']} for "
                     f"width={r['hidden_width']}, shift={r['shift_magnitude']}")
    if not (0.0 <= r['brier_mean'] <= 2.0):
        errors.append(f"Brier score out of range: {r['brier_mean']} for "
                     f"width={r['hidden_width']}, shift={r['shift_magnitude']}")
    if r['n_seeds'] <= 0:
        errors.append(
            f"Invalid seed count: n_seeds={r['n_seeds']} for "
            f"width={r['hidden_width']}, shift={r['shift_magnitude']}"
        )
    if r['ece_std'] < 0.0 or r['accuracy_std'] < 0.0 or r['brier_std'] < 0.0:
        errors.append(
            f"Negative std detected for width={r['hidden_width']}, "
            f"shift={r['shift_magnitude']}"
        )

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
