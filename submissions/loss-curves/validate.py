"""Validate analysis results for completeness and correctness.

Must be run from the submissions/loss-curves/ directory.
"""

import json
import os
import sys

errors = []

# --- Check results files exist ---
for fname in ["results.json", "full_curves.json", "report.txt"]:
    path = os.path.join("results", fname)
    if not os.path.isfile(path):
        errors.append(f"Missing file: {path}")

# --- Check plots exist ---
for fname in [
    "loss_curves_with_fits.png",
    "aic_comparison.png",
    "exponent_distributions.png",
]:
    path = os.path.join("results", fname)
    if not os.path.isfile(path):
        errors.append(f"Missing plot: {path}")

# --- Validate results.json content ---
results_path = os.path.join("results", "results.json")
if os.path.isfile(results_path):
    with open(results_path) as f:
        data = json.load(f)

    # Check metadata
    meta = data.get("metadata", {})
    total_runs = meta.get("total_runs", 0)
    print(f"Total runs: {total_runs}")

    tasks = meta.get("tasks", [])
    print(f"Tasks: {', '.join(tasks)}")
    if not tasks:
        errors.append("Metadata.tasks is empty")

    hidden_sizes = meta.get("hidden_sizes", [])
    print(f"Hidden sizes: {hidden_sizes}")
    if not hidden_sizes:
        errors.append("Metadata.hidden_sizes is empty")

    expected_total = len(tasks) * len(hidden_sizes)
    if total_runs != expected_total:
        errors.append(
            f"total_runs mismatch: expected {expected_total}, got {total_runs}"
        )

    provenance = meta.get("provenance", {})
    required_provenance_fields = [
        "seed",
        "python_version",
        "torch_version",
        "numpy_version",
        "scipy_version",
        "matplotlib_version",
    ]
    for field in required_provenance_fields:
        if provenance.get(field) in (None, ""):
            errors.append(f"Missing metadata.provenance.{field}")
    if provenance:
        print(
            "Provenance: "
            f"py={provenance.get('python_version')} "
            f"torch={provenance.get('torch_version')} "
            f"seed={provenance.get('seed')}"
        )

    # Check runs
    runs = data.get("runs_summary", [])
    print(f"Run summaries: {len(runs)}")
    if len(runs) != total_runs:
        errors.append(f"Expected {total_runs} run summaries, got {len(runs)}")

    # Check each run has fits
    for run in runs:
        if not run.get("fits"):
            errors.append(
                f"Run {run['task']}/h={run['hidden_size']} has no fits"
            )
        else:
            n_converged = sum(1 for f in run["fits"] if f.get("converged"))
            if n_converged < 2:
                errors.append(
                    f"Run {run['task']}/h={run['hidden_size']}: "
                    f"only {n_converged} converged fits (expected >= 2)"
                )
        if run.get("final_loss") is None:
            errors.append(
                f"Run {run['task']}/h={run['hidden_size']}: missing final_loss"
            )
        support = run.get("fit_support")
        if not support:
            errors.append(
                f"Run {run['task']}/h={run['hidden_size']}: missing fit_support"
            )
        else:
            level = support.get("support_level")
            if level not in {"strong", "moderate", "weak", "undetermined"}:
                errors.append(
                    f"Run {run['task']}/h={run['hidden_size']}: invalid support "
                    f"level {level}"
                )
            delta_aic = support.get("delta_aic")
            if delta_aic is not None and delta_aic < 0:
                errors.append(
                    f"Run {run['task']}/h={run['hidden_size']}: "
                    f"negative delta_aic {delta_aic}"
                )

    # Check universality
    uni = data.get("universality", {})
    if not uni.get("majority_form"):
        errors.append("Missing majority_form in universality analysis")
    else:
        print(f"Majority form: {uni['majority_form']} "
              f"({uni.get('majority_fraction', 0):.0%})")

    if not uni.get("form_counts"):
        errors.append("Missing form_counts in universality analysis")

    if not uni.get("best_form_by_task"):
        errors.append("Missing best_form_by_task in universality analysis")
    else:
        for task, form in uni["best_form_by_task"].items():
            print(f"  {task}: best = {form}")

    support_counts = uni.get("support_counts")
    if not support_counts:
        errors.append("Missing support_counts in universality analysis")
    else:
        expected_levels = {"strong", "moderate", "weak", "undetermined"}
        missing = expected_levels.difference(support_counts.keys())
        if missing:
            errors.append(
                "support_counts missing level(s): " + ", ".join(sorted(missing))
            )
        total_support = sum(support_counts.get(level, 0) for level in expected_levels)
        if total_support != total_runs:
            errors.append(
                f"support_counts total mismatch: expected {total_runs}, "
                f"got {total_support}"
            )
        print(
            "Support levels: "
            + ", ".join(f"{k}={support_counts.get(k, 0)}" for k in sorted(expected_levels))
        )

    # Check exponents exist
    if not uni.get("exponents_by_form"):
        errors.append("Missing exponents_by_form")

    elapsed = meta.get("elapsed_seconds", 0)
    print(f"\nRuntime: {elapsed}s")

# --- Report ---
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
