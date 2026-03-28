"""Validate experiment results for completeness and scientific soundness."""

from __future__ import annotations

import json
import sys
from typing import Dict, List

BASELINE_MIN_ACCURACY = 0.5


def _expected_simulation_count(meta: Dict) -> int:
    n_algos = len(meta["algorithms"])
    n_strats = len(meta["strategies"])
    n_sybil_counts = len(meta["sybil_counts"])
    n_seeds = len(meta["seeds"])

    # Expected: for K=0, strategy is "none" so 1 strategy x n_algos x n_seeds
    # For K>0: n_strats strategies x n_algos x n_seeds per K value
    k_zero = 1
    k_nonzero = n_sybil_counts - k_zero
    return n_algos * n_seeds * k_zero + n_algos * n_strats * n_seeds * k_nonzero


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate_results(data: Dict) -> List[str]:
    """Return a list of validation errors for results data."""
    meta = data["metadata"]
    results = data["results"]

    n_algos = len(meta["algorithms"])
    n_strats = len(meta["strategies"])
    expected = _expected_simulation_count(meta)
    actual = len(results)

    errors: List[str] = []

    if actual != expected:
        errors.append(f"Expected {expected} simulations, got {actual}")

    if n_algos < 4:
        errors.append(f"Expected 4 algorithms, got {n_algos}")

    if n_strats < 3:
        errors.append(f"Expected 3 strategies, got {n_strats}")

    # Check metric ranges
    for r in results:
        cfg = r["config"]
        m = r["metrics"]
        label = (
            f"{cfg['algorithm']}/K={cfg['n_sybil']}/"
            f"{cfg['strategy']}/seed={cfg['seed']}"
        )

        acc = m["reputation_accuracy"]
        if not (-1.0 <= acc <= 1.0):
            errors.append(f"{label}: accuracy {acc} out of [-1, 1]")

        det = m["sybil_detection_rate"]
        if not (0.0 <= det <= 1.0):
            errors.append(f"{label}: detection_rate {det} out of [0, 1]")

        wel = m["honest_welfare"]
        if not (0.0 <= wel <= 1.0):
            errors.append(f"{label}: welfare {wel} out of [0, 1]")

        eff = m["market_efficiency"]
        if not (0.0 <= eff <= 1.0):
            errors.append(f"{label}: efficiency {eff} out of [0, 1]")

    baseline_accs = [
        r["metrics"]["reputation_accuracy"]
        for r in results
        if r["config"]["n_sybil"] == 0
    ]
    if baseline_accs:
        mean_baseline = _mean(baseline_accs)
        if mean_baseline < BASELINE_MIN_ACCURACY:
            errors.append(
                "Baseline accuracy too low: "
                f"{mean_baseline:.3f} (expected > {BASELINE_MIN_ACCURACY:.1f})"
            )

    # Check that Sybil attacks degrade simple_average at K=20.
    simple_k20 = [
        r["metrics"]["reputation_accuracy"]
        for r in results
        if r["config"]["algorithm"] == "simple_average"
        and r["config"]["n_sybil"] == 20
    ]
    if simple_k20 and baseline_accs:
        mean_k20 = _mean(simple_k20)
        mean_baseline = _mean(baseline_accs)
        if mean_k20 >= mean_baseline:
            errors.append(
                "Simple average at K=20 should be below baseline: "
                f"K=20={mean_k20:.3f}, baseline={mean_baseline:.3f}"
            )

    return errors


def main() -> int:
    path = "results/results.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]

    with open(path) as f:
        data = json.load(f)

    meta = data["metadata"]
    results = data["results"]

    n_algos = len(meta["algorithms"])
    n_strats = len(meta["strategies"])
    n_sybil_counts = len(meta["sybil_counts"])
    n_seeds = len(meta["seeds"])
    expected = _expected_simulation_count(meta)
    actual = len(results)

    print(f"Algorithms:        {n_algos} ({', '.join(meta['algorithms'])})")
    print(f"Strategies:        {n_strats} ({', '.join(meta['strategies'])})")
    print(f"Sybil counts:      {n_sybil_counts} ({meta['sybil_counts']})")
    print(f"Seeds:             {n_seeds} ({meta['seeds']})")
    print(f"Simulations:       {actual} (expected {expected})")
    print(f"Runtime:           {meta['elapsed_seconds']}s")

    baseline_accs = [
        r["metrics"]["reputation_accuracy"]
        for r in results
        if r["config"]["n_sybil"] == 0
    ]
    if baseline_accs:
        mean_baseline = _mean(baseline_accs)
        print(
            "Baseline accuracy: "
            f"{mean_baseline:.3f} (expected > {BASELINE_MIN_ACCURACY:.1f})"
        )

    simple_k20 = [
        r["metrics"]["reputation_accuracy"]
        for r in results
        if r["config"]["algorithm"] == "simple_average"
        and r["config"]["n_sybil"] == 20
    ]
    if simple_k20 and baseline_accs:
        mean_k20 = _mean(simple_k20)
        print(f"Simple avg K=20:   {mean_k20:.3f} (should be < baseline)")

    errors = validate_results(data)
    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
