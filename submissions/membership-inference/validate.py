#!/usr/bin/env python3
"""Validation script for membership inference scaling results.

Checks that results were produced correctly and meet expected criteria.
Must be run from submissions/membership-inference/ directory.
"""

import json
import os
import sys


def check_working_directory() -> None:
    """Verify we are running from the correct directory."""
    if not os.path.isfile("SKILL.md"):
        print(
            "ERROR: validate.py must be run from submissions/membership-inference/",
            file=sys.stderr,
        )
        print(
            f"Current directory: {os.getcwd()}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    check_working_directory()

    errors: list[str] = []

    # Check results.json exists
    results_path = "results/results.json"
    if not os.path.isfile(results_path):
        errors.append(f"Missing {results_path}")
        print(f"FAIL: {errors[-1]}")
        print(f"\nValidation failed with {len(errors)} error(s).")
        sys.exit(1)

    # Load results
    try:
        with open(results_path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON in {results_path}: {e}")
        print(f"FAIL: {errors[-1]}")
        print(f"\nValidation failed with {len(errors)} error(s).")
        sys.exit(1)

    # Check structure
    if "results" not in data:
        errors.append("Missing 'results' key in results.json")
    if "correlations" not in data:
        errors.append("Missing 'correlations' key in results.json")
    if "config" not in data:
        errors.append("Missing 'config' key in results.json")

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print(f"\nValidation failed with {len(errors)} error(s).")
        sys.exit(1)

    results = data["results"]
    correlations = data["correlations"]
    config = data["config"]

    # Check expected widths
    expected_widths = [16, 32, 64, 128, 256]
    actual_widths = [r["hidden_width"] for r in results]
    if actual_widths != expected_widths:
        errors.append(
            f"Expected widths {expected_widths}, got {actual_widths}"
        )

    # Check each width has required fields
    required_fields = [
        "hidden_width", "n_params", "mean_attack_auc", "std_attack_auc",
        "mean_attack_accuracy", "std_attack_accuracy",
        "mean_overfit_gap", "std_overfit_gap",
        "mean_train_acc", "mean_test_acc", "repeats",
    ]
    for r in results:
        w = r.get("hidden_width", "?")
        for field in required_fields:
            if field not in r:
                errors.append(f"Width {w}: missing field '{field}'")

    # Check AUC values are in valid range [0, 1]
    for r in results:
        w = r["hidden_width"]
        auc = r["mean_attack_auc"]
        if not (0.0 <= auc <= 1.0):
            errors.append(f"Width {w}: attack AUC {auc} out of range [0, 1]")

    # Check that at least some attack succeeds above random (AUC > 0.5)
    max_auc = max(r["mean_attack_auc"] for r in results)
    if max_auc <= 0.5:
        errors.append(
            f"No model width achieved attack AUC > 0.5 (max={max_auc:.3f}). "
            "Attack may not be working."
        )

    # Check correlations exist
    expected_corr_keys = [
        "auc_vs_log_params", "auc_vs_overfit_gap", "gap_vs_log_params"
    ]
    for key in expected_corr_keys:
        if key not in correlations:
            errors.append(f"Missing correlation: {key}")
        else:
            corr = correlations[key]
            if "r" not in corr or "p" not in corr:
                errors.append(f"Correlation {key}: missing r or p value")

    # Check repeats
    for r in results:
        w = r["hidden_width"]
        n_repeats = len(r.get("repeats", []))
        if n_repeats != 3:
            errors.append(f"Width {w}: expected 3 repeats, got {n_repeats}")

    # Check report.md exists
    report_path = "results/report.md"
    if not os.path.isfile(report_path):
        errors.append(f"Missing {report_path}")

    # Check plots exist (optional, warn only)
    plot_files = [
        "results/attack_auc_vs_size.png",
        "results/attack_auc_vs_gap.png",
        "results/overfit_gap_vs_size.png",
        "results/summary_plots.png",
    ]
    missing_plots = [p for p in plot_files if not os.path.isfile(p)]

    # Print summary
    print("Membership Inference Scaling - Validation")
    print("=" * 50)
    print(f"  Model widths tested: {len(results)}")
    print(f"  Repeats per width:   {len(results[0].get('repeats', []))}")
    print(f"  Seed:                {config.get('seed', '?')}")
    print()

    print("Results summary:")
    for r in results:
        print(
            f"  w={r['hidden_width']:3d}  params={r['n_params']:5d}  "
            f"AUC={r['mean_attack_auc']:.3f}+/-{r['std_attack_auc']:.3f}  "
            f"Gap={r['mean_overfit_gap']:.3f}+/-{r['std_overfit_gap']:.3f}"
        )
    print()

    print("Correlations:")
    for key in expected_corr_keys:
        if key in correlations:
            c = correlations[key]
            sig = "sig" if c["p"] < 0.05 else "n.s."
            print(f"  {c['description']}: r={c['r']:.4f}, p={c['p']:.4f} ({sig})")
    print()

    if missing_plots:
        print(f"WARNING: {len(missing_plots)} plot(s) missing (matplotlib may not be installed)")
        for p in missing_plots:
            print(f"  - {p}")
        print()

    if errors:
        for e in errors:
            print(f"FAIL: {e}")
        print(f"\nValidation failed with {len(errors)} error(s).")
        sys.exit(1)
    else:
        print("Validation passed.")


if __name__ == "__main__":
    main()
