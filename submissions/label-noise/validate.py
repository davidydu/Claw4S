#!/usr/bin/env python3
"""Validate results of the label-noise tolerance experiment.

Usage (from submissions/label-noise/):
    .venv/bin/python validate.py

Checks:
    1. Required output files exist
    2. raw_results.json has expected structure and run count
    3. summary.json has expected keys and value ranges
    4. Plots exist and have non-zero size
    5. Scientific sanity checks (accuracy ranges, noise effect direction)

Exit code 0 = all checks pass, 1 = failure.
"""

import json
import os
import sys


def main() -> None:
    # ---- working-directory guard ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    results_dir = "results"
    errors: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # Check 1: Required files exist
    # ------------------------------------------------------------------
    required_files = [
        "raw_results.json",
        "summary.json",
        "arch_sweep.png",
        "width_sweep.png",
    ]
    for fname in required_files:
        path = os.path.join(results_dir, fname)
        if not os.path.isfile(path):
            errors.append(f"Missing required file: {path}")
        elif os.path.getsize(path) == 0:
            errors.append(f"File is empty: {path}")

    if errors:
        _report(errors, warnings)
        return

    # ------------------------------------------------------------------
    # Check 2: raw_results.json structure
    # ------------------------------------------------------------------
    with open(os.path.join(results_dir, "raw_results.json")) as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        errors.append("raw_results.json is not a list")
    else:
        # Architecture sweep: 7 noise * 3 arch * 3 seeds = 63
        # Width sweep: 7 noise * 5 widths * 3 seeds = 105
        # Total: 168
        expected_min = 63 + 105  # 168
        if len(raw) < expected_min:
            errors.append(
                f"raw_results.json has {len(raw)} runs, expected >= {expected_min}"
            )

        # Check fields in first entry
        required_fields = [
            "arch", "depth", "width", "n_params",
            "noise_frac", "seed", "train_acc", "test_acc", "gen_gap",
        ]
        if raw:
            missing = [f for f in required_fields if f not in raw[0]]
            if missing:
                errors.append(f"raw_results.json entries missing fields: {missing}")

        # Check value ranges
        for i, r in enumerate(raw):
            if not (0.0 <= r.get("test_acc", -1) <= 1.0):
                errors.append(f"Run {i}: test_acc={r.get('test_acc')} out of [0,1]")
                break
            if not (0.0 <= r.get("train_acc", -1) <= 1.0):
                errors.append(f"Run {i}: train_acc={r.get('train_acc')} out of [0,1]")
                break
            if not (0.0 <= r.get("noise_frac", -1) <= 1.0):
                errors.append(f"Run {i}: noise_frac={r.get('noise_frac')} out of [0,1]")
                break

    # ------------------------------------------------------------------
    # Check 3: summary.json structure
    # ------------------------------------------------------------------
    with open(os.path.join(results_dir, "summary.json")) as f:
        summary = json.load(f)

    for top_key in ["architecture_sweep", "width_sweep", "findings"]:
        if top_key not in summary:
            errors.append(f"summary.json missing key: {top_key}")

    # Architecture sweep should have 3 architectures
    arch_sweep = summary.get("architecture_sweep", {})
    expected_archs = {"shallow-wide", "medium", "deep-narrow"}
    actual_archs = set(arch_sweep.keys())
    if actual_archs != expected_archs:
        errors.append(
            f"architecture_sweep has {actual_archs}, expected {expected_archs}"
        )

    # Width sweep should have 5 widths
    width_sweep = summary.get("width_sweep", {})
    if len(width_sweep) < 5:
        errors.append(f"width_sweep has {len(width_sweep)} entries, expected 5")

    # ------------------------------------------------------------------
    # Check 4: Plot files have reasonable size (> 5 KB)
    # ------------------------------------------------------------------
    for pname in ["arch_sweep.png", "width_sweep.png"]:
        path = os.path.join(results_dir, pname)
        if os.path.isfile(path):
            size_kb = os.path.getsize(path) / 1024
            if size_kb < 5:
                warnings.append(f"{pname} is only {size_kb:.1f} KB (suspiciously small)")

    # ------------------------------------------------------------------
    # Check 5: Scientific sanity
    # ------------------------------------------------------------------
    # At 0% noise, test accuracy should be above chance (1/5 = 0.20)
    for arch in arch_sweep:
        clean_acc = arch_sweep[arch].get("0%", {}).get("test_acc_mean", 0)
        if clean_acc < 0.30:
            warnings.append(
                f"{arch} at 0% noise has test_acc={clean_acc:.3f}, "
                f"barely above chance (0.20)"
            )

    # Noise should generally hurt: 50% noise accuracy <= 0% noise accuracy
    for arch in arch_sweep:
        clean = arch_sweep[arch].get("0%", {}).get("test_acc_mean", 0)
        noisy = arch_sweep[arch].get("50%", {}).get("test_acc_mean", 0)
        if noisy > clean + 0.05:  # allow small variance
            warnings.append(
                f"{arch}: 50% noise ({noisy:.3f}) > 0% noise ({clean:.3f}) — unexpected"
            )

    # Findings list should not be empty
    findings = summary.get("findings", [])
    if len(findings) < 3:
        warnings.append(f"Only {len(findings)} findings derived, expected >= 3")

    # Each summary entry should have n_runs == 3
    for arch in arch_sweep:
        for nk, stats in arch_sweep[arch].items():
            if stats.get("n_runs", 0) != 3:
                warnings.append(
                    f"arch_sweep[{arch}][{nk}] has n_runs={stats.get('n_runs')}, expected 3"
                )

    _report(errors, warnings)


def _report(errors: list[str], warnings: list[str]) -> None:
    """Print validation report and exit."""
    print("=" * 60)
    print("VALIDATION REPORT")
    print("=" * 60)

    if warnings:
        print(f"\nWARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  [!] {w}")

    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for e in errors:
            print(f"  [X] {e}")
        print("\nRESULT: FAIL")
        sys.exit(1)
    else:
        print(f"\nAll checks passed. ({len(warnings)} warnings)")
        print("RESULT: PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
