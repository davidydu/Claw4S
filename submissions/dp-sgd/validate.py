#!/usr/bin/env python3
"""Validate DP-SGD experiment results.

Checks that:
1. results.json exists and has expected structure
2. DP and baseline run counts match the declared config grid
3. Every (sigma, C, seed) configuration appears exactly once
4. Epsilon values are consistent with noise multiplier ordering
5. Privacy cliff analysis is internally consistent
6. Plots were generated
7. Reproducibility metadata and runtime are recorded

Exit code 0 = all checks pass, 1 = validation failure.

Usage:
    .venv/bin/python validate.py
"""

import json
import os
import sys
from collections import defaultdict

# ── Working-directory guard ──────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")

FALLBACK_EXPECTED_DP_RUNS = 63
FALLBACK_EXPECTED_BASELINE_RUNS = 3
EXPECTED_RUNTIME_MAX_SECONDS = 180.0
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
    all_passed &= check("config" in results, "results.json has 'config' key")
    all_passed &= check("metadata" in results, "results.json has 'metadata' key")
    all_passed &= check(
        "baseline_runs" in results, "results.json has 'baseline_runs' key"
    )
    all_passed &= check("dp_runs" in results, "results.json has 'dp_runs' key")

    config = results.get("config", {})
    metadata = results.get("metadata", {})
    baseline_runs = results.get("baseline_runs", [])
    dp_runs = results.get("dp_runs", [])

    noise_multipliers = config.get("noise_multipliers", [])
    clipping_norms = config.get("clipping_norms", [])
    seeds = config.get("seeds", [])

    config_lists_ok = (
        isinstance(noise_multipliers, list)
        and isinstance(clipping_norms, list)
        and isinstance(seeds, list)
        and len(noise_multipliers) > 0
        and len(clipping_norms) > 0
        and len(seeds) > 0
    )
    all_passed &= check(
        config_lists_ok,
        "Config contains non-empty lists for noise_multipliers, clipping_norms, seeds",
    )

    if config_lists_ok:
        expected_dp_runs = len(noise_multipliers) * len(clipping_norms) * len(seeds)
        expected_baseline_runs = len(seeds)
    else:
        expected_dp_runs = FALLBACK_EXPECTED_DP_RUNS
        expected_baseline_runs = FALLBACK_EXPECTED_BASELINE_RUNS

    # ── Check 3: correct number of runs ──────────────────────────────
    n_baseline = len(baseline_runs)
    n_dp = len(dp_runs)
    all_passed &= check(
        n_baseline == expected_baseline_runs,
        f"Baseline runs: {n_baseline} (expected {expected_baseline_runs})",
    )
    all_passed &= check(
        n_dp == expected_dp_runs,
        f"DP runs: {n_dp} (expected {expected_dp_runs})",
    )

    # ── Check 4: all runs have required fields ───────────────────────
    required_fields = {"accuracy", "epsilon", "noise_multiplier", "max_norm", "seed"}
    fields_ok = True
    for i, run in enumerate(dp_runs):
        missing = required_fields - set(run.keys())
        if missing:
            fields_ok = False
            print(f"    Run {i} missing fields: {missing}")
    all_passed &= check(fields_ok, "All DP runs have required fields")

    # ── Check 5: full config coverage exactly once ───────────────────
    if config_lists_ok:
        expected_triplets = {
            (float(sigma), float(C), int(seed))
            for sigma in noise_multipliers
            for C in clipping_norms
            for seed in seeds
        }
        actual_triplets = {
            (float(run["noise_multiplier"]), float(run["max_norm"]), int(run["seed"]))
            for run in dp_runs
        }
        all_passed &= check(
            len(actual_triplets) == len(dp_runs),
            "No duplicate (sigma, C, seed) combinations in DP runs",
        )
        missing_triplets = expected_triplets - actual_triplets
        unexpected_triplets = actual_triplets - expected_triplets
        all_passed &= check(
            len(missing_triplets) == 0,
            f"All expected DP configurations present ({len(expected_triplets)} total)",
        )
        all_passed &= check(
            len(unexpected_triplets) == 0,
            "No unexpected DP configurations present",
        )

    # ── Check 6: accuracy in valid range ─────────────────────────────
    if dp_runs:
        accs = [r["accuracy"] for r in dp_runs]
        all_passed &= check(
            all(0.0 <= a <= 1.0 for a in accs),
            f"All accuracies in [0, 1]: min={min(accs):.4f}, max={max(accs):.4f}",
        )
    else:
        all_passed &= check(False, "DP runs are non-empty")

    # ── Check 7: epsilon ordering ────────────────────────────────────
    # Higher noise_multiplier should give lower epsilon (more private)
    groups: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for run in dp_runs:
        key = (run["max_norm"], run["seed"])
        groups[key].append(run)

    ordering_ok = True
    for key, runs in groups.items():
        sorted_runs = sorted(runs, key=lambda r: r["noise_multiplier"])
        epsilons = [r["epsilon"] for r in sorted_runs]
        for i in range(len(epsilons) - 1):
            if epsilons[i] < epsilons[i + 1]:
                ordering_ok = False
                C, seed = key
                print(
                    f"    Ordering violation at C={C}, seed={seed}: "
                    f"sigma={sorted_runs[i]['noise_multiplier']} (eps={epsilons[i]:.2f}) < "
                    f"sigma={sorted_runs[i + 1]['noise_multiplier']} (eps={epsilons[i + 1]:.2f})"
                )

    all_passed &= check(
        ordering_ok,
        "Epsilon decreases as noise_multiplier increases (monotonicity)",
    )

    # ── Check 8: baseline sanity + privacy-utility trend ─────────────
    if baseline_runs:
        baseline_accs = [r["accuracy"] for r in baseline_runs]
        baseline_mean = sum(baseline_accs) / len(baseline_accs)
        all_passed &= check(
            baseline_mean >= 0.5,
            f"Baseline accuracy reasonable: {baseline_mean:.4f} (>= 0.50)",
        )

        high_noise_accs = [
            r["accuracy"] for r in dp_runs if r["noise_multiplier"] >= 5.0
        ]
        low_noise_accs = [
            r["accuracy"] for r in dp_runs if r["noise_multiplier"] <= 0.1
        ]

        if high_noise_accs and low_noise_accs:
            high_noise_mean = sum(high_noise_accs) / len(high_noise_accs)
            low_noise_mean = sum(low_noise_accs) / len(low_noise_accs)
            all_passed &= check(
                low_noise_mean > high_noise_mean,
                f"Privacy-utility tradeoff: low-noise acc ({low_noise_mean:.4f}) > "
                f"high-noise acc ({high_noise_mean:.4f})",
            )
    else:
        all_passed &= check(False, "Baseline runs are non-empty")

    # ── Check 9: expected seeds are used ─────────────────────────────
    seeds_used_dp = {int(r["seed"]) for r in dp_runs}
    seeds_used_baseline = {int(r["seed"]) for r in baseline_runs}
    expected_seeds = {int(s) for s in seeds} if config_lists_ok else set()

    if expected_seeds:
        all_passed &= check(
            seeds_used_dp == expected_seeds,
            f"DP runs cover expected seeds: {sorted(seeds_used_dp)}",
        )
        all_passed &= check(
            seeds_used_baseline == expected_seeds,
            f"Baseline runs cover expected seeds: {sorted(seeds_used_baseline)}",
        )
    else:
        all_passed &= check(
            len(seeds_used_dp) >= 3,
            f"Multiple seeds used in DP runs: {len(seeds_used_dp)} (>= 3)",
        )

    # ── Check 10: metadata includes reproducibility + runtime ────────
    elapsed_seconds = metadata.get("elapsed_seconds")
    all_passed &= check(
        isinstance(elapsed_seconds, (int, float)),
        "metadata.elapsed_seconds is present",
    )
    if isinstance(elapsed_seconds, (int, float)):
        all_passed &= check(
            elapsed_seconds <= EXPECTED_RUNTIME_MAX_SECONDS,
            f"Runtime <= {EXPECTED_RUNTIME_MAX_SECONDS:.0f}s ({elapsed_seconds:.2f}s)",
        )

    reproducibility = metadata.get("reproducibility")
    required_repro_fields = {
        "python_version",
        "python_implementation",
        "torch_version",
        "numpy_version",
        "torch_deterministic_algorithms_enabled",
        "torch_num_threads",
    }
    all_passed &= check(
        isinstance(reproducibility, dict),
        "metadata.reproducibility exists",
    )
    if isinstance(reproducibility, dict):
        all_passed &= check(
            required_repro_fields.issubset(set(reproducibility)),
            "metadata.reproducibility has required fields",
        )
        all_passed &= check(
            reproducibility.get("torch_deterministic_algorithms_enabled") is True,
            "Deterministic torch algorithms enabled",
        )
        threads = reproducibility.get("torch_num_threads")
        all_passed &= check(
            isinstance(threads, int) and threads >= 1,
            f"Torch thread count valid ({threads})",
        )

    # ── Check 11: plots exist ────────────────────────────────────────
    for plot_name in EXPECTED_PLOTS:
        plot_path = os.path.join(RESULTS_DIR, plot_name)
        all_passed &= check(os.path.isfile(plot_path), f"Plot exists: {plot_name}")

    # ── Check 12: summary.json has privacy cliff analysis ────────────
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.isfile(summary_path):
        with open(summary_path, "r") as f:
            summary = json.load(f)
        all_passed &= check(
            "privacy_cliff" in summary,
            "summary.json contains privacy cliff analysis",
        )
        all_passed &= check(
            "summaries" in summary,
            "summary.json contains configuration summaries",
        )
        privacy_cliff = summary.get("privacy_cliff", {})
        if privacy_cliff.get("n_configs_below_threshold") == 0:
            all_passed &= check(
                privacy_cliff.get("cliff_epsilon") is None,
                "No cliff epsilon reported when no configuration collapses",
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
