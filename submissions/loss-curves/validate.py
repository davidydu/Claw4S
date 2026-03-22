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
    if total_runs != 12:
        errors.append(f"Expected 12 runs, got {total_runs}")

    tasks = meta.get("tasks", [])
    print(f"Tasks: {', '.join(tasks)}")
    if len(tasks) != 4:
        errors.append(f"Expected 4 tasks, got {len(tasks)}")

    hidden_sizes = meta.get("hidden_sizes", [])
    print(f"Hidden sizes: {hidden_sizes}")
    if hidden_sizes != [32, 64, 128]:
        errors.append(f"Expected hidden_sizes [32, 64, 128], got {hidden_sizes}")

    # Check runs
    runs = data.get("runs_summary", [])
    print(f"Run summaries: {len(runs)}")
    if len(runs) != 12:
        errors.append(f"Expected 12 run summaries, got {len(runs)}")

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
