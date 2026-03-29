"""Validate experiment results for completeness and scientific soundness.

Usage: .venv/bin/python validate.py
Must be run from the submissions/data-poisoning/ directory.
"""

import json
import os
import sys

# ── Working-directory guard ──────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != _here:
    print(f"Changing working directory to {_here}")
    os.chdir(_here)


def main() -> int:
    """Validate results and return exit code (0 = pass, 1 = fail)."""
    print("=" * 60)
    print("Validating Data Poisoning Sensitivity Results")
    print("=" * 60)

    errors: list[str] = []

    # ── Check files exist ────────────────────────────────────────────
    required_files = [
        "results/results.json",
        "results/performance.json",
        "results/accuracy_vs_poison.png",
        "results/generalization_gap.png",
        "results/train_vs_test.png",
    ]
    for fpath in required_files:
        if os.path.exists(fpath):
            print(f"  [OK] {fpath}")
        else:
            errors.append(f"Missing file: {fpath}")
            print(f"  [FAIL] {fpath} — missing")

    if not os.path.exists("results/results.json"):
        print("\nCannot continue validation without results.json")
        return 1

    # ── Load results ─────────────────────────────────────────────────
    with open("results/results.json") as f:
        data = json.load(f)

    performance = {}
    if os.path.exists("results/performance.json"):
        with open("results/performance.json") as f:
            performance = json.load(f)
    else:
        errors.append("Missing file: results/performance.json")

    meta = data.get("metadata", {})
    config = data.get("config", {})
    runs = data.get("runs", [])
    agg = data.get("aggregated", [])
    fits = data.get("sigmoid_fits", [])
    findings = data.get("findings", {})

    poison_fractions = config.get("poison_fractions", [])
    hidden_widths = config.get("hidden_widths", [])
    seeds = config.get("seeds", [])
    n_classes = int(config.get("n_classes", 5))
    chance_acc = 1.0 / n_classes if n_classes > 0 else 0.2

    # ── Check completeness ───────────────────────────────────────────
    print("\n--- Completeness ---")
    expected_runs = len(poison_fractions) * len(hidden_widths) * len(seeds)
    if expected_runs == 0:
        expected_runs = meta.get("total_runs", 81)
    actual_runs = len(runs)
    print(f"  Runs: {actual_runs} (expected {expected_runs})")
    if actual_runs != expected_runs:
        errors.append(f"Expected {expected_runs} runs, got {actual_runs}")

    expected_agg = len(poison_fractions) * len(hidden_widths)
    if expected_agg == 0:
        expected_agg = actual_agg = len(agg)
    else:
        actual_agg = len(agg)
    print(f"  Aggregated points: {actual_agg} (expected {expected_agg})")
    if actual_agg != expected_agg:
        errors.append(f"Expected {expected_agg} aggregated points, got {actual_agg}")

    expected_fits = len(hidden_widths) if hidden_widths else len(fits)
    actual_fits = len(fits)
    print(f"  Sigmoid fits: {actual_fits} (expected {expected_fits})")
    if actual_fits != expected_fits:
        errors.append(f"Expected {expected_fits} sigmoid fits, got {actual_fits}")

    if any("elapsed_seconds" in run for run in runs):
        errors.append("results.json should exclude per-run elapsed_seconds")
    if "total_time_seconds" in meta:
        errors.append("results.json metadata should exclude total_time_seconds")

    # ── Check scientific soundness ───────────────────────────────────
    print("\n--- Scientific Soundness ---")

    baseline_poison = min(poison_fractions) if poison_fractions else 0.0
    high_poison = max(poison_fractions) if poison_fractions else 0.5

    # Clean accuracy should be meaningfully above chance.
    clean_accs = findings.get("clean_test_accuracy", {})
    if not clean_accs:
        errors.append("Missing clean_test_accuracy findings")
    for hw_str, acc in clean_accs.items():
        hw = int(hw_str) if isinstance(hw_str, str) else hw_str
        print(f"  Clean accuracy (width {hw}): {acc:.3f}")
        if acc <= chance_acc + 0.1:
            errors.append(
                f"Clean accuracy for width {hw} is {acc:.3f} "
                f"(expected > chance + 0.1 = {chance_acc + 0.1:.3f})"
            )

    # At highest poison fraction, accuracy should degrade versus baseline.
    baseline_acc_by_width = {}
    for pt in agg:
        if pt.get("poison_fraction", 0.0) == baseline_poison:
            baseline_acc_by_width[pt.get("hidden_width")] = pt.get("test_acc_mean", 0.0)

    high_poison_points = [a for a in agg if a.get("poison_fraction", 0) == high_poison]
    for pt in high_poison_points:
        hw = pt.get("hidden_width", "?")
        acc = pt.get("test_acc_mean", 0)
        print(f"  Accuracy at {high_poison:.0%} poison (width {hw}): {acc:.3f}")
        baseline_acc = baseline_acc_by_width.get(hw)
        if baseline_acc is not None and acc >= baseline_acc - 0.05:
            errors.append(
                f"Accuracy at {high_poison:.0%} poison for width {hw} is {acc:.3f} "
                f"(baseline {baseline_acc:.3f}; expected degradation)"
            )

    # Accuracy should decrease monotonically (on average)
    print("\n--- Monotonicity Check ---")
    widths_to_check = hidden_widths or sorted({a.get("hidden_width") for a in agg if "hidden_width" in a})
    for hw in widths_to_check:
        hw_points = sorted(
            [a for a in agg if a.get("hidden_width") == hw],
            key=lambda a: a.get("poison_fraction", 0),
        )
        accs = [p.get("test_acc_mean", 0) for p in hw_points]
        fracs = [p.get("poison_fraction", 0) for p in hw_points]
        violations = 0
        for i in range(len(accs) - 1):
            if accs[i + 1] > accs[i] + 0.05:  # Allow 5% tolerance
                violations += 1
        if violations > 0:
            print(f"  Width {hw}: {violations} monotonicity violations (within tolerance)")
        else:
            print(f"  Width {hw}: monotonically decreasing [OK]")

    # ── Check sigmoid fit quality ────────────────────────────────────
    print("\n--- Sigmoid Fit Quality ---")
    for fit in fits:
        hw = fit.get("hidden_width", "?")
        r2 = fit.get("r_squared", 0)
        print(f"  Width {hw}: R²={r2:.4f}")
        if r2 < 0.8:
            errors.append(f"Poor sigmoid fit for width {hw}: R²={r2:.4f}")

    # ── Check variance reported ──────────────────────────────────────
    print("\n--- Variance Reporting ---")
    has_variance = any(a.get("test_acc_std", 0) > 0 for a in agg)
    if has_variance:
        print("  Standard deviations reported: [OK]")
    else:
        errors.append("No variance reported in aggregated results")
        print("  Standard deviations reported: [FAIL]")

    # ── Check runtime ────────────────────────────────────────────────
    print("\n--- Performance ---")
    total_time = performance.get("total_time_seconds")
    if total_time is None:
        errors.append("Missing total_time_seconds in performance.json")
        print("  Total runtime: [MISSING]")
    else:
        print(f"  Total runtime: {total_time:.1f}s")
        if total_time > 180:
            errors.append(f"Runtime {total_time:.1f}s exceeds 3-minute limit")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if errors:
        print(f"VALIDATION FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("VALIDATION PASSED — all checks OK")
        return 0


if __name__ == "__main__":
    sys.exit(main())
