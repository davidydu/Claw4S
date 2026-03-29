#!/usr/bin/env python3
"""Validation script for membership inference scaling results.

Checks that results were produced correctly and meet expected criteria.
Must be run from submissions/membership-inference/ directory.
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List


REQUIRED_RESULT_FIELDS = [
    "hidden_width",
    "n_params",
    "mean_attack_auc",
    "std_attack_auc",
    "mean_attack_accuracy",
    "std_attack_accuracy",
    "mean_overfit_gap",
    "std_overfit_gap",
    "mean_train_acc",
    "mean_test_acc",
    "repeats",
]

REQUIRED_REPEAT_FIELDS = [
    "repeat",
    "hidden_width",
    "train_acc",
    "test_acc",
    "overfit_gap",
    "attack_auc",
    "attack_accuracy",
]

EXPECTED_CORR_KEYS = [
    "auc_vs_log_params",
    "auc_vs_overfit_gap",
    "gap_vs_log_params",
]


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


def expected_param_count(hidden_width: int, input_dim: int, output_dim: int) -> int:
    """Expected parameter count for 2-layer MLP (Linear-ReLU-Linear)."""
    return (input_dim * hidden_width + hidden_width) + (
        hidden_width * output_dim + output_dim
    )


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def validate_results_payload(data: Dict[str, Any]) -> List[str]:
    """Validate loaded JSON payload and return all detected errors."""
    errors: List[str] = []

    if "results" not in data:
        errors.append("Missing 'results' key in results.json")
    if "correlations" not in data:
        errors.append("Missing 'correlations' key in results.json")
    if "config" not in data:
        errors.append("Missing 'config' key in results.json")
    if errors:
        return errors

    results = data["results"]
    correlations = data["correlations"]
    config = data["config"]

    if not isinstance(results, list) or not results:
        errors.append("'results' must be a non-empty list")
        return errors
    if not isinstance(correlations, dict):
        errors.append("'correlations' must be a dictionary")
        return errors
    if not isinstance(config, dict):
        errors.append("'config' must be a dictionary")
        return errors

    expected_widths = config.get("hidden_widths")
    if isinstance(expected_widths, list) and expected_widths:
        actual_widths = [r.get("hidden_width") for r in results]
        if actual_widths != expected_widths:
            errors.append(
                f"Expected widths {expected_widths}, got {actual_widths}"
            )

    expected_repeats = _as_int(config.get("n_repeats"), 3)
    n_features = _as_int(config.get("n_features"), 10)
    n_classes = _as_int(config.get("n_classes"), 5)

    for r in results:
        w = r.get("hidden_width", "?")
        for field in REQUIRED_RESULT_FIELDS:
            if field not in r:
                errors.append(f"Width {w}: missing field '{field}'")
        if any(field not in r for field in REQUIRED_RESULT_FIELDS):
            continue

        if not (0.0 <= r["mean_attack_auc"] <= 1.0):
            errors.append(
                f"Width {w}: attack AUC {r['mean_attack_auc']} out of range [0, 1]"
            )
        if not (0.0 <= r["mean_attack_accuracy"] <= 1.0):
            errors.append(
                f"Width {w}: attack accuracy {r['mean_attack_accuracy']} out of range [0, 1]"
            )
        if not (0.0 <= r["mean_train_acc"] <= 1.0):
            errors.append(
                f"Width {w}: train accuracy {r['mean_train_acc']} out of range [0, 1]"
            )
        if not (0.0 <= r["mean_test_acc"] <= 1.0):
            errors.append(
                f"Width {w}: test accuracy {r['mean_test_acc']} out of range [0, 1]"
            )

        expected_params = expected_param_count(
            hidden_width=r["hidden_width"],
            input_dim=n_features,
            output_dim=n_classes,
        )
        if r["n_params"] != expected_params:
            errors.append(
                f"Width {w}: n_params {r['n_params']} does not match expected {expected_params}"
            )

        repeats = r.get("repeats", [])
        if len(repeats) != expected_repeats:
            errors.append(
                f"Width {w}: expected {expected_repeats} repeats, got {len(repeats)}"
            )

        for rep in repeats:
            rep_idx = rep.get("repeat", "?")
            for field in REQUIRED_REPEAT_FIELDS:
                if field not in rep:
                    errors.append(
                        f"Width {w} repeat {rep_idx}: missing field '{field}'"
                    )
            if any(field not in rep for field in REQUIRED_REPEAT_FIELDS):
                continue

            if rep["hidden_width"] != r["hidden_width"]:
                errors.append(
                    f"Width {w} repeat {rep_idx}: hidden_width mismatch ({rep['hidden_width']})"
                )
            if not (0.0 <= rep["attack_auc"] <= 1.0):
                errors.append(
                    f"Width {w} repeat {rep_idx}: attack_auc {rep['attack_auc']} out of range [0, 1]"
                )
            if not (0.0 <= rep["attack_accuracy"] <= 1.0):
                errors.append(
                    f"Width {w} repeat {rep_idx}: attack_accuracy {rep['attack_accuracy']} out of range [0, 1]"
                )

    max_auc = max((r["mean_attack_auc"] for r in results), default=0.0)
    if max_auc <= 0.5:
        errors.append(
            f"No model width achieved attack AUC > 0.5 (max={max_auc:.3f}). "
            "Attack may not be working."
        )

    for key in EXPECTED_CORR_KEYS:
        if key not in correlations:
            errors.append(f"Missing correlation: {key}")
            continue
        corr = correlations[key]
        if "r" not in corr or "p" not in corr:
            errors.append(f"Correlation {key}: missing r or p value")
            continue
        if not (-1.0 <= corr["r"] <= 1.0):
            errors.append(f"Correlation {key}: r={corr['r']} out of range [-1, 1]")
        if not (0.0 <= corr["p"] <= 1.0):
            errors.append(f"Correlation {key}: p={corr['p']} out of range [0, 1]")

    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate membership-inference experiment outputs."
    )
    parser.add_argument(
        "--results-path",
        default="results/results.json",
        help="Path to results JSON produced by run.py.",
    )
    return parser.parse_args()


def main() -> None:
    check_working_directory()

    args = parse_args()
    errors: List[str] = []

    # Check results.json exists
    results_path = args.results_path
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

    errors.extend(validate_results_payload(data))
    results = data.get("results", [])
    correlations = data.get("correlations", {})
    config = data.get("config", {})

    results_dir = os.path.dirname(results_path) or "."

    # Check report.md exists in same output directory
    report_path = os.path.join(results_dir, "report.md")
    if not os.path.isfile(report_path):
        errors.append(f"Missing {report_path}")

    # Check plots exist in output directory (optional, warn only)
    plot_files = [
        os.path.join(results_dir, "attack_auc_vs_size.png"),
        os.path.join(results_dir, "attack_auc_vs_gap.png"),
        os.path.join(results_dir, "overfit_gap_vs_size.png"),
        os.path.join(results_dir, "summary_plots.png"),
    ]
    missing_plots = [p for p in plot_files if not os.path.isfile(p)]

    # Print summary
    print("Membership Inference Scaling - Validation")
    print("=" * 50)
    print(f"  Model widths tested: {len(results)}")
    print(f"  Repeats per width:   {_as_int(config.get('n_repeats'), 3)}")
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
    for key in EXPECTED_CORR_KEYS:
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
