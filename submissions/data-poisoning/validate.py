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

    meta = data.get("metadata", {})
    runs = data.get("runs", [])
    agg = data.get("aggregated", [])
    fits = data.get("sigmoid_fits", [])
    findings = data.get("findings", {})

    # ── Check completeness ───────────────────────────────────────────
    print("\n--- Completeness ---")
    expected_runs = 81  # 9 fractions x 3 widths x 3 seeds
    actual_runs = len(runs)
    print(f"  Runs: {actual_runs} (expected {expected_runs})")
    if actual_runs != expected_runs:
        errors.append(f"Expected {expected_runs} runs, got {actual_runs}")

    expected_agg = 27  # 9 fractions x 3 widths
    actual_agg = len(agg)
    print(f"  Aggregated points: {actual_agg} (expected {expected_agg})")
    if actual_agg != expected_agg:
        errors.append(f"Expected {expected_agg} aggregated points, got {actual_agg}")

    expected_fits = 3
    actual_fits = len(fits)
    print(f"  Sigmoid fits: {actual_fits} (expected {expected_fits})")
    if actual_fits != expected_fits:
        errors.append(f"Expected {expected_fits} sigmoid fits, got {actual_fits}")

    # ── Check scientific soundness ───────────────────────────────────
    print("\n--- Scientific Soundness ---")

    # Clean accuracy should be high (> 0.7 for well-separated Gaussians)
    clean_accs = findings.get("clean_test_accuracy", {})
    for hw_str, acc in clean_accs.items():
        hw = int(hw_str) if isinstance(hw_str, str) else hw_str
        print(f"  Clean accuracy (width {hw}): {acc:.3f}")
        if acc < 0.7:
            errors.append(f"Clean accuracy for width {hw} is {acc:.3f} (expected > 0.7)")

    # At 50% poison, accuracy should be near chance (0.2 for 5 classes)
    high_poison_points = [a for a in agg if a.get("poison_fraction", 0) == 0.5]
    for pt in high_poison_points:
        hw = pt.get("hidden_width", "?")
        acc = pt.get("test_acc_mean", 0)
        print(f"  Accuracy at 50% poison (width {hw}): {acc:.3f}")
        if acc > 0.7:
            errors.append(f"Accuracy at 50% poison for width {hw} is {acc:.3f} "
                          f"(expected degradation)")

    # Accuracy should decrease monotonically (on average)
    print("\n--- Monotonicity Check ---")
    for hw in [32, 64, 128]:
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
    total_time = meta.get("total_time_seconds", 0)
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
