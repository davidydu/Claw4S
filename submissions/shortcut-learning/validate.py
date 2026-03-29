"""Validate experiment outputs for completeness and reproducibility."""

import json
import os
import sys

from src.validation import collect_validation_errors


def _ensure_submission_cwd(script_name: str) -> None:
    """Ensure script is run from submissions/shortcut-learning/."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.samefile(os.getcwd(), script_dir):
        print(f"ERROR: {script_name} must be executed from submissions/shortcut-learning/")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Expected: {script_dir}")
        sys.exit(1)


def main() -> None:
    _ensure_submission_cwd("validate.py")

    results_path = os.path.join("results", "results.json")
    if not os.path.exists(results_path):
        print(f"ERROR: {results_path} not found. Run run.py first.")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    errors = collect_validation_errors(data)

    meta = data.get("metadata", {})
    print(f"Total configurations: {meta.get('n_configs', 0)}")

    runs = data.get("individual_runs", [])
    print(f"Individual runs: {len(runs)}")

    aggs = data.get("aggregates", [])
    print(f"Aggregate entries: {len(aggs)}")

    no_reg_runs = [run for run in runs if run.get("weight_decay") == 0.0]
    positive_reliance = sum(
        1 for run in no_reg_runs if run.get("shortcut_reliance", 0.0) > 0.0
    )
    print(
        "Unregularized runs with positive shortcut reliance: "
        f"{positive_reliance}/{len(no_reg_runs)}"
    )

    if runs:
        avg_with = sum(run["test_acc_with_shortcut"] for run in runs) / len(runs)
        avg_without = sum(run["test_acc_without_shortcut"] for run in runs) / len(runs)
    else:
        avg_with = 0.0
        avg_without = 0.0
    print(f"Average test acc with shortcut: {avg_with:.4f}")
    print(f"Average test acc without shortcut: {avg_without:.4f}")

    findings = data.get("findings", [])
    print(f"Findings: {len(findings)}")

    report_path = os.path.join("results", "report.md")
    if not os.path.exists(report_path):
        errors.append("results/report.md not found")
    else:
        with open(report_path) as f:
            report = f.read()
        if len(report) < 200:
            errors.append(f"Report too short ({len(report)} chars)")
        print(f"Report length: {len(report)} chars")

    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    print("\nValidation passed.")


if __name__ == "__main__":
    main()
