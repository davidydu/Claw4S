"""Validate experiment outputs for completeness and scientific sanity."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.output_spec import PRIMARY_ARTIFACTS


EXPECTED_SUMMARIES = 3 * 3 * 5 * 3
EXPECTED_DERIVED = 3 * 3 * 3


def _required_artifact_paths(results_dir: Path) -> list[Path]:
    """Map canonical artifact spec to concrete files inside *results_dir*."""
    names = [Path(rel_path).name for rel_path in PRIMARY_ARTIFACTS]
    return [results_dir / name for name in names]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing results.json and report.md (default: results)",
    )
    return parser.parse_args(argv)


def _load_results(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def _validate_payload(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    meta = data.get("metadata", {})
    raw_results = data.get("raw_results", [])
    summaries = data.get("summaries", [])
    derived_metrics = data.get("derived_metrics", [])
    amplifications = data.get("amplifications", [])

    n_configs = meta.get("total_configs")
    n_raw = len(raw_results)
    n_summaries = len(summaries)
    n_derived = len(derived_metrics)

    print(f"Configurations: {n_raw} (expected {n_configs})")
    print(f"Summary groups: {n_summaries}")
    print(f"Derived metrics: {n_derived}")
    print(f"Amplifications: {len(amplifications)}")

    if n_configs is None:
        errors.append("Missing metadata.total_configs")
    elif n_raw != n_configs:
        errors.append(f"Expected {n_configs} raw results, got {n_raw}")

    if n_summaries != EXPECTED_SUMMARIES:
        errors.append(f"Expected {EXPECTED_SUMMARIES} summary groups, got {n_summaries}")

    if n_derived != EXPECTED_DERIVED:
        errors.append(f"Expected {EXPECTED_DERIVED} derived metrics, got {n_derived}")

    baselines = [r for r in raw_results if r.get("byzantine_fraction") == 0.0]
    if not baselines:
        errors.append("No baseline (f=0) results found")
    else:
        avg_baseline = sum(r["accuracy"] for r in baselines) / len(baselines)
        print(f"Baseline accuracy (f=0): {avg_baseline:.3f}")
        if avg_baseline < 0.50:
            errors.append(f"Baseline accuracy {avg_baseline:.3f} unexpectedly low (< 0.50)")

    for s in summaries:
        mean_acc = s.get("mean_accuracy")
        if mean_acc is None or mean_acc < 0.0 or mean_acc > 1.0:
            errors.append(f"Accuracy {mean_acc} out of [0,1] range")

    for d in derived_metrics:
        threshold = d.get("byzantine_threshold_50")
        if threshold is None or threshold < 0.0 or threshold > 1.0:
            errors.append(f"Byzantine threshold {threshold} out of [0,1] range for {d}")

    for d in derived_metrics:
        score = d.get("resilience_score")
        if score is None or score < 0.0 or score > 1.0:
            errors.append(f"Resilience score {score} out of [0,1] range for {d}")

    repro_groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for r in raw_results:
        key = (
            r.get("honest_type"),
            r.get("byzantine_type"),
            r.get("byzantine_fraction"),
            r.get("committee_size"),
            r.get("seed"),
        )
        repro_groups[key].append(r.get("accuracy"))
    for key, accs in repro_groups.items():
        if len(accs) > 1:
            errors.append(f"Duplicate config-seed combo: {key}")

    return errors


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    results_dir = Path(args.results_dir)
    required = _required_artifact_paths(results_dir)
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Missing required artifact(s):")
        for path in missing:
            print(f"  - {path}")
        print("Run `python run.py` from the submission directory to regenerate artifacts.")
        return 1

    data = _load_results(results_dir / "results.json")
    errors = _validate_payload(data)
    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for error in errors:
            print(f"  - {error}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
