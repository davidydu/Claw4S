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

import json
import os
import sys

# --- Working-directory guard ---
_expected_marker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "SKILL.md")
if not os.path.isfile(_expected_marker):
    print("ERROR: validate.py must be executed from the dp-calibration/ "
          "submission directory.", file=sys.stderr)
    sys.exit(1)
os.chdir(os.path.dirname(os.path.abspath(__file__)))
# --------------------------------


def load_results(path: str = "results/results.json") -> dict:
    """Load results JSON, converting 'Infinity' strings back to float."""
    with open(path) as f:
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
    errors = []
    warnings = []

    print("=" * 60)
    print("Validating DP Noise Calibration Results")
    print("=" * 60)
    print()

    # Check 1: results.json exists
    results_path = "results/results.json"
    if not os.path.isfile(results_path):
        print("FAIL: results/results.json not found. Run run.py first.")
        sys.exit(1)

    data = load_results(results_path)
    print("[1/6] Results file loaded successfully.")

    # Check 2: Metadata and grid completeness
    meta = data["metadata"]
    expected_configs = meta["num_T"] * meta["num_delta"] * meta["num_sigma"]
    actual_configs = len(data["results"])
    print(f"[2/6] Grid: {meta['num_T']}T x {meta['num_delta']}d x "
          f"{meta['num_sigma']}s = {expected_configs} configs")
    print(f"      Actual results: {actual_configs}")
    if actual_configs != expected_configs:
        errors.append(f"Expected {expected_configs} configs, got {actual_configs}")

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

    # Check tightness ratios for naive and advanced
    naive_avg = summary["avg_tightness_ratio"].get("naive", 0)
    advanced_avg = summary["avg_tightness_ratio"].get("advanced", 0)
    if isinstance(naive_avg, (int, float)) and naive_avg != float("inf"):
        print(f"      Naive avg tightness: {naive_avg:.2f}x")
    if isinstance(advanced_avg, (int, float)) and advanced_avg != float("inf"):
        print(f"      Advanced avg tightness: {advanced_avg:.2f}x")

    # Check 6: Visualization files
    print("[6/6] Checking visualization files...")
    expected_figs = [
        "results/epsilon_vs_T.png",
        "results/tightness_heatmap.png",
        "results/method_comparison.png",
        "results/epsilon_vs_sigma.png",
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
