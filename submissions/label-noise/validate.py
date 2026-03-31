#!/usr/bin/env python3
"""Validate results of the label-noise tolerance experiment.

Usage (from submissions/label-noise/):
    .venv/bin/python validate.py
    .venv/bin/python validate.py --results-dir results

Checks:
    1. Required output files exist
    2. raw_results.json has expected structure and run count
    3. summary.json has expected keys and value ranges
    4. Plots exist and have non-zero size
    5. Scientific sanity checks (accuracy ranges, noise effect direction)

Exit code 0 = all checks pass, 1 = failure.
"""

import argparse
import json
import math
import os
import sys

NOISE_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
NOISE_KEYS = [f"{n:.0%}" for n in NOISE_FRACS]
SEEDS = [42, 43, 44]

ARCH_CONFIGS = {
    "shallow-wide": (1, 200),
    "medium": (2, 70),
    "deep-narrow": (4, 35),
}
WIDTH_SWEEP_DEPTH = 2
WIDTH_SWEEP_WIDTHS = [16, 32, 64, 128, 256]
WIDTH_SWEEP_NAMES = {f"d{WIDTH_SWEEP_DEPTH}_w{w}" for w in WIDTH_SWEEP_WIDTHS}

EXPECTED_ARCH_RUNS = len(ARCH_CONFIGS) * len(NOISE_FRACS) * len(SEEDS)  # 63
EXPECTED_WIDTH_RUNS = len(WIDTH_SWEEP_WIDTHS) * len(NOISE_FRACS) * len(SEEDS)  # 105
EXPECTED_TOTAL_RUNS = EXPECTED_ARCH_RUNS + EXPECTED_WIDTH_RUNS  # 168


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate output artifacts for the label-noise experiment."
    )
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing raw_results.json, summary.json, and plots.",
    )
    return parser.parse_args(argv)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _match_noise_frac(value: object) -> float | None:
    if not _is_finite_number(value):
        return None
    v = float(value)
    for expected in NOISE_FRACS:
        if abs(v - expected) < 1e-9:
            return expected
    return None


def _format_preview(items: list[tuple], max_items: int = 3) -> str:
    preview = ", ".join(str(x) for x in items[:max_items])
    if len(items) > max_items:
        preview += f", ... (+{len(items) - max_items} more)"
    return preview


def validate_results(results_dir: str = "results") -> tuple[list[str], list[str]]:
    """Validate artifacts in *results_dir* and return (errors, warnings)."""
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
        return errors, warnings

    # ------------------------------------------------------------------
    # Check 2: raw_results.json structure + exact run completeness
    # ------------------------------------------------------------------
    with open(os.path.join(results_dir, "raw_results.json"), encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        errors.append("raw_results.json is not a list")
    else:
        if len(raw) != EXPECTED_TOTAL_RUNS:
            errors.append(
                f"raw_results.json has {len(raw)} runs; expected exactly "
                f"{EXPECTED_TOTAL_RUNS}"
            )

        required_fields = [
            "arch",
            "depth",
            "width",
            "n_params",
            "noise_frac",
            "seed",
            "train_acc",
            "test_acc",
            "gen_gap",
            "wall_seconds",
        ]

        seen_arch_keys: set[tuple[str, float, int]] = set()
        seen_width_keys: set[tuple[int, float, int]] = set()
        arch_rows = 0
        width_rows = 0

        for i, row in enumerate(raw):
            ctx = f"run {i}"
            if not isinstance(row, dict):
                errors.append(f"{ctx}: entry is not a JSON object")
                continue

            missing = [f for f in required_fields if f not in row]
            if missing:
                errors.append(f"{ctx}: missing required fields {missing}")
                continue

            arch = row["arch"]
            depth = row["depth"]
            width = row["width"]
            noise = _match_noise_frac(row["noise_frac"])
            seed = row["seed"]

            if not isinstance(arch, str):
                errors.append(f"{ctx}: arch must be a string, got {type(arch).__name__}")
                continue
            if not isinstance(depth, int):
                errors.append(f"{ctx}: depth must be int, got {type(depth).__name__}")
                continue
            if not isinstance(width, int):
                errors.append(f"{ctx}: width must be int, got {type(width).__name__}")
                continue
            if noise is None:
                errors.append(f"{ctx}: noise_frac={row['noise_frac']} not in {NOISE_FRACS}")
                continue
            if not isinstance(seed, int):
                errors.append(f"{ctx}: seed must be int, got {type(seed).__name__}")
                continue
            if seed not in SEEDS:
                errors.append(f"{ctx}: seed={seed} not in expected seeds {SEEDS}")

            for metric_name in ["train_acc", "test_acc"]:
                metric = row[metric_name]
                if not _is_finite_number(metric):
                    errors.append(f"{ctx}: {metric_name} is not a finite number")
                elif not (0.0 <= float(metric) <= 1.0):
                    errors.append(
                        f"{ctx}: {metric_name}={metric} out of expected range [0,1]"
                    )

            gap = row["gen_gap"]
            if not _is_finite_number(gap):
                errors.append(f"{ctx}: gen_gap is not a finite number")
            elif not (-1.0 <= float(gap) <= 1.0):
                errors.append(f"{ctx}: gen_gap={gap} out of expected range [-1,1]")

            wall = row["wall_seconds"]
            if not _is_finite_number(wall):
                errors.append(f"{ctx}: wall_seconds is not a finite number")
            elif float(wall) <= 0:
                errors.append(f"{ctx}: wall_seconds={wall} must be > 0")

            if arch in ARCH_CONFIGS:
                exp_depth, exp_width = ARCH_CONFIGS[arch]
                if depth != exp_depth or width != exp_width:
                    errors.append(
                        f"{ctx}: architecture '{arch}' expected "
                        f"(depth={exp_depth}, width={exp_width}) but got "
                        f"(depth={depth}, width={width})"
                    )
                key = (arch, noise, seed)
                if key in seen_arch_keys:
                    errors.append(f"{ctx}: duplicate run detected for {key}")
                seen_arch_keys.add(key)
                arch_rows += 1
            elif arch in WIDTH_SWEEP_NAMES:
                arch_prefix = f"d{WIDTH_SWEEP_DEPTH}_w"
                width_from_name = int(arch.removeprefix(arch_prefix))
                if width_from_name not in WIDTH_SWEEP_WIDTHS:
                    errors.append(
                        f"{ctx}: width sweep run has unsupported width {width_from_name}"
                    )
                if depth != WIDTH_SWEEP_DEPTH or width != width_from_name:
                    errors.append(
                        f"{ctx}: width sweep run '{arch}' expected "
                        f"(depth={WIDTH_SWEEP_DEPTH}, width={width_from_name}) but got "
                        f"(depth={depth}, width={width})"
                    )
                key = (width_from_name, noise, seed)
                if key in seen_width_keys:
                    errors.append(f"{ctx}: duplicate run detected for width key {key}")
                seen_width_keys.add(key)
                width_rows += 1
            else:
                errors.append(
                    f"{ctx}: unknown arch identifier '{arch}' "
                    f"(expected one of {sorted(ARCH_CONFIGS)} "
                    f"or width IDs {sorted(WIDTH_SWEEP_NAMES)})"
                )

        if arch_rows != EXPECTED_ARCH_RUNS:
            errors.append(
                f"Architecture sweep rows={arch_rows}; expected {EXPECTED_ARCH_RUNS}"
            )
        if width_rows != EXPECTED_WIDTH_RUNS:
            errors.append(f"Width sweep rows={width_rows}; expected {EXPECTED_WIDTH_RUNS}")

        expected_arch_keys = {
            (arch, noise, seed)
            for arch in ARCH_CONFIGS
            for noise in NOISE_FRACS
            for seed in SEEDS
        }
        expected_width_keys = {
            (width, noise, seed)
            for width in WIDTH_SWEEP_WIDTHS
            for noise in NOISE_FRACS
            for seed in SEEDS
        }

        missing_arch_keys = sorted(expected_arch_keys - seen_arch_keys)
        if missing_arch_keys:
            errors.append(
                "Missing expected run(s) in architecture sweep: "
                f"{_format_preview(missing_arch_keys)}"
            )

        missing_width_keys = sorted(expected_width_keys - seen_width_keys)
        if missing_width_keys:
            errors.append(
                "Missing expected run(s) in width sweep: "
                f"{_format_preview(missing_width_keys)}"
            )

    # ------------------------------------------------------------------
    # Check 3: summary.json structure
    # ------------------------------------------------------------------
    with open(os.path.join(results_dir, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    if not isinstance(summary, dict):
        errors.append("summary.json is not a JSON object")
        return errors, warnings

    for top_key in ["architecture_sweep", "width_sweep", "findings"]:
        if top_key not in summary:
            errors.append(f"summary.json missing key: {top_key}")

    arch_sweep = summary.get("architecture_sweep", {})
    width_sweep = summary.get("width_sweep", {})

    if not isinstance(arch_sweep, dict):
        errors.append("architecture_sweep must be an object")
        arch_sweep = {}
    if not isinstance(width_sweep, dict):
        errors.append("width_sweep must be an object")
        width_sweep = {}

    expected_archs = set(ARCH_CONFIGS.keys())
    actual_archs = set(arch_sweep.keys())
    if actual_archs != expected_archs:
        errors.append(
            f"architecture_sweep has {actual_archs}, expected {expected_archs}"
        )

    expected_width_names = WIDTH_SWEEP_NAMES
    actual_width_names = set(width_sweep.keys())
    if actual_width_names != expected_width_names:
        errors.append(
            f"width_sweep has {actual_width_names}, expected {expected_width_names}"
        )

    summary_fields = [
        "test_acc_mean",
        "test_acc_std",
        "train_acc_mean",
        "train_acc_std",
        "gen_gap_mean",
        "gen_gap_std",
        "n_runs",
    ]

    for arch in expected_archs:
        noise_stats = arch_sweep.get(arch, {})
        if not isinstance(noise_stats, dict):
            errors.append(f"architecture_sweep[{arch}] must be an object")
            continue
        noise_keys = set(noise_stats.keys())
        if noise_keys != set(NOISE_KEYS):
            errors.append(
                f"architecture_sweep[{arch}] has noise keys {noise_keys}, "
                f"expected {set(NOISE_KEYS)}"
            )
        for nk in NOISE_KEYS:
            stats = noise_stats.get(nk)
            if not isinstance(stats, dict):
                errors.append(f"architecture_sweep[{arch}][{nk}] must be an object")
                continue
            for field in summary_fields:
                if field not in stats:
                    errors.append(f"architecture_sweep[{arch}][{nk}] missing {field}")
            if stats.get("n_runs") != 3:
                errors.append(
                    f"architecture_sweep[{arch}][{nk}] has n_runs={stats.get('n_runs')}, "
                    "expected 3"
                )

    for wname in expected_width_names:
        noise_stats = width_sweep.get(wname, {})
        if not isinstance(noise_stats, dict):
            errors.append(f"width_sweep[{wname}] must be an object")
            continue
        noise_keys = set(noise_stats.keys())
        if noise_keys != set(NOISE_KEYS):
            errors.append(
                f"width_sweep[{wname}] has noise keys {noise_keys}, "
                f"expected {set(NOISE_KEYS)}"
            )
        for nk in NOISE_KEYS:
            stats = noise_stats.get(nk)
            if not isinstance(stats, dict):
                errors.append(f"width_sweep[{wname}][{nk}] must be an object")
                continue
            for field in summary_fields:
                if field not in stats:
                    errors.append(f"width_sweep[{wname}][{nk}] missing {field}")
            if stats.get("n_runs") != 3:
                errors.append(
                    f"width_sweep[{wname}][{nk}] has n_runs={stats.get('n_runs')}, "
                    "expected 3"
                )

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
    for arch in expected_archs:
        clean_acc = arch_sweep.get(arch, {}).get("0%", {}).get("test_acc_mean")
        if _is_finite_number(clean_acc) and float(clean_acc) < 0.30:
            warnings.append(
                f"{arch} at 0% noise has test_acc={float(clean_acc):.3f}, "
                "barely above chance (0.20)"
            )

    for arch in expected_archs:
        clean = arch_sweep.get(arch, {}).get("0%", {}).get("test_acc_mean")
        noisy = arch_sweep.get(arch, {}).get("50%", {}).get("test_acc_mean")
        if _is_finite_number(clean) and _is_finite_number(noisy):
            if float(noisy) > float(clean) + 0.05:
                warnings.append(
                    f"{arch}: 50% noise ({float(noisy):.3f}) > "
                    f"0% noise ({float(clean):.3f}) — unexpected"
                )

    findings = summary.get("findings", [])
    if not isinstance(findings, list):
        errors.append("summary.findings must be a list")
    elif len(findings) < 3:
        warnings.append(f"Only {len(findings)} findings derived, expected >= 3")

    return errors, warnings


def main() -> None:
    # ---- working-directory guard ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    args = _parse_args()
    errors, warnings = validate_results(results_dir=args.results_dir)
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
