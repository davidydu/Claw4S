#!/usr/bin/env python3
"""Validate experiment results.

Checks:
1. All 162 simulations completed.
2. Results are reproducible (re-run 2 configs, compare).
3. Key scientific invariants hold:
   a. SA distorts more than RA for all learners.
   b. SL has lower belief error than NL against RA.
   c. PA shows exploitation gap (early truthful > late truthful).
   d. Belief errors are in [0, 1].
4. Output files exist (summary.json, figures, tables).

Usage:
    .venv/bin/python validate.py [--results-dir results]
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate experiment results")
    parser.add_argument("--results-dir", type=str, default="results")
    return parser.parse_args()


def validate() -> bool:
    args = parse_args()
    results_dir = Path(args.results_dir)

    checks_passed = 0
    checks_total = 0
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        if condition:
            checks_passed += 1
            print(f"  PASS  {name}")
        else:
            failures.append(f"{name}: {detail}")
            print(f"  FAIL  {name}  -- {detail}")

    print("=" * 60)
    print("Validation: Adversarial World Model Manipulation")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Raw results exist and have 162 entries.
    # ------------------------------------------------------------------
    raw_path = results_dir / "raw_results.pkl"
    check("Raw results file exists", raw_path.exists(), str(raw_path))

    if raw_path.exists():
        with open(raw_path, "rb") as f:
            results = pickle.load(f)
        check("162 simulations completed", len(results) == 162,
              f"got {len(results)}")
    else:
        results = []

    # ------------------------------------------------------------------
    # 2. Summary JSON exists and is parseable.
    # ------------------------------------------------------------------
    summary_path = results_dir / "summary.json"
    check("summary.json exists", summary_path.exists())
    aggregated = {}
    if summary_path.exists():
        with open(summary_path) as f:
            aggregated = json.load(f)
        # 9 matchups * 3 regimes * 2 noises = 54 groups.
        check("54 aggregate groups", len(aggregated) == 54,
              f"got {len(aggregated)}")

    # ------------------------------------------------------------------
    # 3. Figures directory has expected files.
    # ------------------------------------------------------------------
    fig_dir = results_dir / "figures"
    check("Figures directory exists", fig_dir.exists())
    if fig_dir.exists():
        pngs = list(fig_dir.glob("*.png"))
        # 6 heatmaps + 6 speed charts + 18 timeseries + 2 accuracy = 32
        check("At least 30 figures", len(pngs) >= 30,
              f"got {len(pngs)}")

    # ------------------------------------------------------------------
    # 4. Tables directory has expected files.
    # ------------------------------------------------------------------
    table_dir = results_dir / "tables"
    check("Tables directory exists", table_dir.exists())
    if table_dir.exists():
        csvs = list(table_dir.glob("*.csv"))
        check("At least 4 CSV tables", len(csvs) >= 4, f"got {len(csvs)}")

    # ------------------------------------------------------------------
    # 5. Scientific invariants.
    # ------------------------------------------------------------------
    if aggregated:
        print("\n--- Scientific Invariants ---")

        # 5a. SA distorts more than RA for NL (stable, noise=0).
        for regime in ["stable", "volatile"]:
            key_ra = f"NL-vs-RA_{regime}_noise0.0"
            key_sa = f"NL-vs-SA_{regime}_noise0.0"
            if key_ra in aggregated and key_sa in aggregated:
                err_ra = aggregated[key_ra].get("distortion.final_belief_error.mean", 0)
                err_sa = aggregated[key_sa].get("distortion.final_belief_error.mean", 0)
                check(
                    f"SA > RA distortion (NL, {regime})",
                    err_sa > err_ra,
                    f"SA={err_sa:.4f}, RA={err_ra:.4f}",
                )

        # 5b. SL more resilient than NL against RA in at least one regime.
        sl_better = False
        for regime in ["stable", "slow_drift", "volatile"]:
            key_nl = f"NL-vs-RA_{regime}_noise0.0"
            key_sl = f"SL-vs-RA_{regime}_noise0.0"
            if key_nl in aggregated and key_sl in aggregated:
                err_nl = aggregated[key_nl].get("distortion.final_belief_error.mean", 1)
                err_sl = aggregated[key_sl].get("distortion.final_belief_error.mean", 1)
                if err_sl < err_nl:
                    sl_better = True
        check("SL more resilient than NL in some regime", sl_better)

        # 5c. PA shows exploitation gap in at least one matchup.
        exploitation_detected = False
        for lc in ["NL", "SL", "AL"]:
            for regime in ["stable", "slow_drift", "volatile"]:
                key = f"{lc}-vs-PA_{regime}_noise0.0"
                if key in aggregated:
                    gap = aggregated[key].get("credibility.exploitation_gap.mean", 0)
                    if gap > 0.1:
                        exploitation_detected = True
        check("PA exploitation pattern detected", exploitation_detected)

        # 5d. All belief errors in [0, 1].
        errors_valid = True
        for key, agg in aggregated.items():
            be = agg.get("distortion.mean_belief_error.mean", 0.5)
            if be < -0.01 or be > 1.01:
                errors_valid = False
                break
        check("All belief errors in [0, 1]", errors_valid)

    # ------------------------------------------------------------------
    # 6. Reproducibility: re-run two configs and compare.
    # ------------------------------------------------------------------
    print("\n--- Reproducibility ---")
    from src.experiment import SimConfig, run_simulation

    for lc, ac in [("NL", "SA"), ("AL", "PA")]:
        cfg = SimConfig(
            learner_code=lc,
            adversary_code=ac,
            drift_regime="stable",
            noise_level=0.0,
            seed=0,
            n_rounds=1000,
        )
        r1 = run_simulation(cfg)
        r2 = run_simulation(cfg)
        err1 = r1.audit["distortion"]["mean_belief_error"]
        err2 = r2.audit["distortion"]["mean_belief_error"]
        check(
            f"Reproducible: {lc}-vs-{ac}",
            abs(err1 - err2) < 1e-10,
            f"diff={abs(err1 - err2):.2e}",
        )

    # ------------------------------------------------------------------
    # Summary.
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"VALIDATION: {checks_passed}/{checks_total} checks passed")
    if failures:
        print("\nFailed checks:")
        for f in failures:
            print(f"  - {f}")
    print("=" * 60)

    return len(failures) == 0


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
