"""Validate DP noise calibration results for completeness and correctness.

Checks:
1. results.json exists and has expected structure
2. All grid points are present (72 configurations)
3. All four methods have finite epsilon for reasonable parameters
4. Tightness ratios are >= 1.0 (sanity check)
5. GDP or RDP wins most configurations (scientific finding)
6. Visualization files exist

Must be run from the submission directory:
    .venv/bin/python validate.py
"""

import argparse
import json
import os
import sys
from importlib import metadata as importlib_metadata

from src.analysis import (
    DELTA_VALUES,
    SIGMA_VALUES,
    T_VALUES,
    compute_results_digest,
)

# --- Working-directory guard ---
_expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "SKILL.md")
if not os.path.isfile(_expected_marker):
    print("ERROR: validate.py must be executed from the dp-calibration/ "
          "submission directory.", file=sys.stderr)
    sys.exit(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --------------------------------

# Expected reproducibility fingerprint for the default pinned grid.
# Updated when formulas/grid intentionally change.
EXPECTED_PINNED_RESULTS_DIGEST = "1d93cec82a3e3e76bb62a347d178fc25ca1a609b9329b1843ebe533b21c70217"


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for validation runs."""
    parser = argparse.ArgumentParser(
        description="Validate dp-calibration results.json and generated figures."
    )
    parser.add_argument(
        "--results-path",
        default="results/results.json",
        help="Path to JSON output from run.py",
    )
    parser.add_argument(
        "--skip-pinned-check",
        action="store_true",
        help="Disable pinned-grid digest + win-count checks",
    )
    return parser.parse_args()


def load_results(path: str) -> dict:
    """Load results JSON, converting 'Infinity' strings back to float."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    def _convert(obj):
        if isinstance(obj, str):
            if obj == "Infinity":
                return float("inf")
            if obj == "-Infinity":
                return float("-inf")
            if obj == "NaN":
                return float("nan")
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    return _convert(data)


def main():
    args = _parse_args()

    errors = []
    warnings = []

    print("=" * 60)
    print("Validating DP Noise Calibration Results")
    print("=" * 60)
    print()

    # Check 1: results.json exists
    results_path = args.results_path
    if not os.path.isfile(results_path):
        print(f"FAIL: {results_path} not found. Run run.py first.")
        sys.exit(1)

    data = load_results(results_path)
    print("[1/6] Results file loaded successfully.")

    required_top_keys = {"metadata", "grid", "results", "summary"}
    missing_top = sorted(required_top_keys - set(data.keys()))
    if missing_top:
        errors.append(f"Missing top-level keys: {missing_top}")
        print(f"      Missing keys: {missing_top}")

    # Check 2: Metadata and grid completeness
    meta = data["metadata"]
    expected_configs = meta["num_T"] * meta["num_delta"] * meta["num_sigma"]
    actual_configs = len(data["results"])
    print(f"[2/6] Grid: {meta['num_T']}T x {meta['num_delta']}d x "
          f"{meta['num_sigma']}s = {expected_configs} configs")
    print(f"      Actual results: {actual_configs}")
    if actual_configs != expected_configs:
        errors.append(f"Expected {expected_configs} configs, got {actual_configs}")

    # Reproducibility metadata checks
    print("      Verifying reproducibility metadata...")
    digest = meta.get("results_digest")
    if not isinstance(digest, str) or len(digest) != 64:
        errors.append("metadata.results_digest missing or malformed")
    else:
        computed_digest = compute_results_digest(data["results"])
        if computed_digest != digest:
            errors.append(
                f"results_digest mismatch: metadata={digest}, computed={computed_digest}"
            )

    package_versions = meta.get("package_versions")
    if not isinstance(package_versions, dict):
        errors.append("metadata.package_versions missing")
    else:
        for pkg_name, recorded_version in package_versions.items():
            try:
                runtime_version = importlib_metadata.version(pkg_name)
            except importlib_metadata.PackageNotFoundError:
                errors.append(f"Runtime package missing: {pkg_name}")
                continue
            if runtime_version != recorded_version:
                errors.append(
                    f"Package version mismatch for {pkg_name}: "
                    f"results={recorded_version}, runtime={runtime_version}"
                )

    # Check 3: All methods produce finite epsilon for sigma >= 1.0
    print("[3/6] Checking finite epsilon for sigma >= 1.0...")
    methods = data["grid"]["methods"]
    for r in data["results"]:
        if r["sigma"] >= 1.0:
            for m in methods:
                eps = r["epsilons"].get(m, float("inf"))
                if eps == float("inf") or eps == "Infinity":
                    errors.append(
                        f"Infinite epsilon for {m} at T={r['T']}, "
                        f"sigma={r['sigma']}, delta={r['delta']}")

    if not errors:
        print("      All finite — OK")

    # Check 4: Tightness ratios >= 1.0
    print("[4/6] Checking tightness ratios >= 1.0...")
    ratio_violations = 0
    for r in data["results"]:
        for m, ratio in r["tightness_ratio"].items():
            if isinstance(ratio, (int, float)) and ratio < 0.999:
                ratio_violations += 1
                errors.append(
                    f"Tightness ratio < 1.0 for {m} at T={r['T']}, "
                    f"sigma={r['sigma']}: {ratio:.4f}")
    if ratio_violations == 0:
        print("      All ratios >= 1.0 — OK")

    # Check 5: Scientific findings
    print("[5/6] Checking scientific findings...")
    summary = data["summary"]
    wins = summary["win_counts"]
    total = sum(wins.values())

    gdp_wins = wins.get("gdp", 0)
    rdp_wins = wins.get("rdp", 0)
    tight_wins = gdp_wins + rdp_wins
    tight_pct = 100 * tight_wins / total if total > 0 else 0

    print(f"      GDP wins: {gdp_wins} ({100*gdp_wins/total:.1f}%)")
    print(f"      RDP wins: {rdp_wins} ({100*rdp_wins/total:.1f}%)")
    print(f"      GDP+RDP wins: {tight_wins} ({tight_pct:.1f}%)")

    if tight_pct < 50:
        warnings.append(
            f"GDP+RDP win only {tight_pct:.1f}% of configs "
            "(expected > 50%)")

    # Robustness metrics are present and internally consistent
    median_tightness = summary.get("median_tightness_ratio", {})
    p95_tightness = summary.get("p95_tightness_ratio", {})
    for method in data["grid"]["methods"]:
        if method not in median_tightness or method not in p95_tightness:
            errors.append(
                f"Missing robust summary stats for {method} "
                "(median/p95 tightness)"
            )
            continue
        median = median_tightness[method]
        p95 = p95_tightness[method]
        if isinstance(median, (int, float)) and median < 0.999:
            errors.append(f"Median tightness < 1.0 for {method}: {median:.4f}")
        if (
            isinstance(median, (int, float))
            and isinstance(p95, (int, float))
            and p95 + 1e-9 < median
        ):
            errors.append(
                f"p95 tightness < median for {method}: p95={p95:.4f}, "
                f"median={median:.4f}"
            )

    # Pinned-grid reproducibility checks (optional skip)
    is_pinned_grid = (
        data["grid"]["T_values"] == T_VALUES
        and data["grid"]["delta_values"] == DELTA_VALUES
        and data["grid"]["sigma_values"] == SIGMA_VALUES
    )
    if is_pinned_grid and not args.skip_pinned_check:
        expected_wins = {"naive": 7, "advanced": 0, "rdp": 0, "gdp": 65}
        if wins != expected_wins:
            errors.append(f"Pinned-grid wins changed: expected {expected_wins}, got {wins}")
        if EXPECTED_PINNED_RESULTS_DIGEST == "TO_BE_FILLED":
            errors.append("Pinned digest constant not initialized in validate.py")
        elif digest != EXPECTED_PINNED_RESULTS_DIGEST:
            errors.append(
                f"Pinned-grid digest changed: expected {EXPECTED_PINNED_RESULTS_DIGEST}, "
                f"got {digest}"
            )
    elif is_pinned_grid:
        print("      Pinned-grid checks skipped (--skip-pinned-check).")
    else:
        print("      Custom grid detected; pinned-grid checks not applied.")

    # Check tightness ratios for naive and advanced
    naive_avg = summary["avg_tightness_ratio"].get("naive", 0)
    advanced_avg = summary["avg_tightness_ratio"].get("advanced", 0)
    if isinstance(naive_avg, (int, float)) and naive_avg != float("inf"):
        print(f"      Naive avg tightness: {naive_avg:.2f}x")
    if isinstance(advanced_avg, (int, float)) and advanced_avg != float("inf"):
        print(f"      Advanced avg tightness: {advanced_avg:.2f}x")

    # Check 6: Visualization files
    print("[6/6] Checking visualization files...")
    results_dir = os.path.dirname(results_path) or "."
    expected_figs = [
        os.path.join(results_dir, "epsilon_vs_T.png"),
        os.path.join(results_dir, "tightness_heatmap.png"),
        os.path.join(results_dir, "method_comparison.png"),
        os.path.join(results_dir, "epsilon_vs_sigma.png"),
    ]
    for fig in expected_figs:
        if os.path.isfile(fig):
            size_kb = os.path.getsize(fig) / 1024
            print(f"      {fig}: {size_kb:.0f} KB — OK")
        else:
            errors.append(f"Missing figure: {fig}")

    # Final verdict
    print()
    print("=" * 60)
    if errors:
        print(f"FAIL: {len(errors)} error(s)")
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)
    elif warnings:
        print(f"PASS with {len(warnings)} warning(s)")
        for w in warnings:
            print(f"  WARNING: {w}")
    else:
        print("PASS: All checks passed")

    print()
    print(f"Runtime: {meta['elapsed_seconds']:.3f}s")
    print(f"Configurations: {actual_configs}")
    print(f"Computations: {meta['total_computations']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
