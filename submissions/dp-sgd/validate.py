#!/usr/bin/env python3
"""Validate DP-SGD experiment results.

Checks that:
1. results.json exists and has expected structure
2. All 63 DP runs + 3 baseline runs are present
3. Epsilon values are consistent with noise multiplier ordering
4. Privacy cliff analysis is internally consistent
5. Plots were generated
6. Statistical variance is reported (multiple seeds)

Exit code 0 = all checks pass, 1 = validation failure.

Usage:
    .venv/bin/python validate.py
"""

import json
import os
import sys

# ── Working-directory guard ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

# Expected counts
EXPECTED_DP_RUNS = 63       # 7 sigma x 3 C x 3 seeds
EXPECTED_BASELINE_RUNS = 3  # 3 seeds
EXPECTED_PLOTS = [
    "privacy_utility_curve.png",
    "utility_gap.png",
    "clipping_effect.png",
    "summary.json",
]


def check(condition: bool, msg: str) -> bool:
    """Print pass/fail for a check and return result."""
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {msg}")
    return condition


def main() -> None:
    """Run all validation checks."""
    print("=" * 60)
    print("DP-SGD Results Validation")
    print("=" * 60)

    all_passed = True

    # ── Check 1: results.json exists ─────────────────────────────────
    results_path = os.path.join(RESULTS_DIR, "results.json")
    if not check(os.path.isfile(results_path), "results.json exists"):
        print("\nFATAL: results.json not found. Run run.py first.")
        sys.exit(1)

    with open(results_path, "r") as f:
        results = json.load(f)

    # ── Check 2: expected structure ──────────────────────────────────
    all_passed &= check(
        "config" in results, "results.json has 'config' key"
    )
    all_passed &= check(
        "baseline_runs" in results, "results.json has 'baseline_runs' key"
    )
    all_passed &= check(
        "dp_runs" in results, "results.json has 'dp_runs' key"
    )

    # ── Check 3: correct number of runs ──────────────────────────────
    n_baseline = len(results.get("baseline_runs", []))
    n_dp = len(results.get("dp_runs", []))
    all_passed &= check(
        n_baseline == EXPECTED_BASELINE_RUNS,
        f"Baseline runs: {n_baseline} (expected {EXPECTED_BASELINE_RUNS})"
    )
    all_passed &= check(
        n_dp == EXPECTED_DP_RUNS,
        f"DP runs: {n_dp} (expected {EXPECTED_DP_RUNS})"
    )

    # ── Check 4: all runs have required fields ───────────────────────
    required_fields = {"accuracy", "epsilon", "noise_multiplier", "max_norm", "seed"}
    dp_runs = results.get("dp_runs", [])

    fields_ok = True
    for i, run in enumerate(dp_runs):
        missing = required_fields - set(run.keys())
        if missing:
            fields_ok = False
            print(f"    Run {i} missing fields: {missing}")
    all_passed &= check(fields_ok, "All DP runs have required fields")

    # ── Check 5: accuracy in valid range ─────────────────────────────
    accs = [r["accuracy"] for r in dp_runs]
    all_passed &= check(
        all(0.0 <= a <= 1.0 for a in accs),
        f"All accuracies in [0, 1]: min={min(accs):.4f}, max={max(accs):.4f}"
    )

    # ── Check 6: epsilon ordering ────────────────────────────────────
    # Higher noise_multiplier should give lower epsilon (more private)
    # Group by (max_norm, seed), check epsilon ordering within groups
    from collections import defaultdict
    groups: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for run in dp_runs:
        key = (run["max_norm"], run["seed"])
        groups[key].append(run)

    ordering_ok = True
    for key, runs in groups.items():
        sorted_runs = sorted(runs, key=lambda r: r["noise_multiplier"])
        epsilons = [r["epsilon"] for r in sorted_runs]
        # Epsilon should decrease as noise_multiplier increases
        for i in range(len(epsilons) - 1):
            if epsilons[i] < epsilons[i + 1]:
                ordering_ok = False
                C, seed = key
                print(
                    f"    Ordering violation at C={C}, seed={seed}: "
                    f"sigma={sorted_runs[i]['noise_multiplier']} (eps={epsilons[i]:.2f}) < "
                    f"sigma={sorted_runs[i+1]['noise_multiplier']} (eps={epsilons[i+1]:.2f})"
                )

    all_passed &= check(
        ordering_ok,
        "Epsilon decreases as noise_multiplier increases (monotonicity)"
    )

    # ── Check 7: baseline accuracy is reasonable ─────────────────────
    baseline_runs = results.get("baseline_runs", [])
    if baseline_runs:
        baseline_accs = [r["accuracy"] for r in baseline_runs]
        baseline_mean = sum(baseline_accs) / len(baseline_accs)
        all_passed &= check(
            baseline_mean >= 0.5,
            f"Baseline accuracy reasonable: {baseline_mean:.4f} (>= 0.50)"
        )

        # ── Check 8: privacy cliff pattern ───────────────────────────
        # At least some high-noise configs should have much lower accuracy
        high_noise_accs = [
            r["accuracy"] for r in dp_runs
            if r["noise_multiplier"] >= 5.0
        ]
        low_noise_accs = [
            r["accuracy"] for r in dp_runs
            if r["noise_multiplier"] <= 0.1
        ]

        if high_noise_accs and low_noise_accs:
            high_noise_mean = sum(high_noise_accs) / len(high_noise_accs)
            low_noise_mean = sum(low_noise_accs) / len(low_noise_accs)
            all_passed &= check(
                low_noise_mean > high_noise_mean,
                f"Privacy-utility tradeoff: low-noise acc ({low_noise_mean:.4f}) > "
                f"high-noise acc ({high_noise_mean:.4f})"
            )

    # ── Check 9: variance reported (multiple seeds) ──────────────────
    seeds_used = set(r["seed"] for r in dp_runs)
    all_passed &= check(
        len(seeds_used) >= 3,
        f"Multiple seeds used: {len(seeds_used)} (>= 3)"
    )

    # ── Check 10: plots exist ────────────────────────────────────────
    for plot_name in EXPECTED_PLOTS:
        plot_path = os.path.join(RESULTS_DIR, plot_name)
        all_passed &= check(
            os.path.isfile(plot_path),
            f"Plot exists: {plot_name}"
        )

    # ── Check 11: summary.json has privacy cliff analysis ────────────
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        all_passed &= check(
            "privacy_cliff" in summary,
            "summary.json contains privacy cliff analysis"
        )
        all_passed &= check(
            "summaries" in summary,
            "summary.json contains configuration summaries"
        )
        privacy_cliff = summary.get("privacy_cliff", {})
        if privacy_cliff.get("n_configs_below_threshold") == 0:
            all_passed &= check(
                privacy_cliff.get("cliff_epsilon") is None,
                "No cliff epsilon reported when no configuration collapses"
            )

    # ── Check 12: runtime was reasonable ─────────────────────────────
    elapsed = results.get("elapsed_seconds", 999)
    all_passed &= check(
        elapsed <= 180,
        f"Runtime: {elapsed:.1f}s (<= 180s)"
    )

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    if all_passed:
        print("VALIDATION PASSED: All checks passed.")
    else:
        print("VALIDATION FAILED: Some checks failed.")
    print("=" * 60)

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
