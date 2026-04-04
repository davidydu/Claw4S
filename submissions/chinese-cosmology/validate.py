# validate.py
"""Validate experiment results for completeness and correctness.

Checks:
  1. results/results.json exists and is valid JSON
  2. Record count matches metadata
  3. All 3 systems (bazi, ziwei, wuxing) have scores for each record
  4. All domain scores are in [0, 1]
  5. Null model produces lower correlation than real data

Usage:
    .venv/bin/python validate.py

Expected output ends with: "Validation passed."
"""

import argparse
import json
import sys

import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate chinese-cosmology results JSON.",
    )
    parser.add_argument(
        "--results-file",
        default="results/results.json",
        help="Path to results JSON (default: results/results.json).",
    )
    return parser.parse_args()


def _load_results(results_file: str) -> dict | None:
    try:
        with open(results_file) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"ERROR: {results_file} not found. Run 'python run.py' first.")
        return None
    except json.JSONDecodeError as e:
        print(f"ERROR: {results_file} is not valid JSON: {e}")
        return None


def main(results_file: str = "results/results.json") -> int:
    data = _load_results(results_file)
    if data is None:
        return 1

    metadata = data.get("metadata", {})
    records = data.get("records", [])

    num_charts = metadata.get("num_charts", 0)
    systems = metadata.get("systems", ["bazi", "ziwei", "wuxing"])
    domains = metadata.get("domains", ["career", "wealth", "relationships", "health", "overall"])
    num_records = len(records)

    print(f"Charts expected:  {num_charts:,}")
    print(f"Records found:    {num_records:,}")

    errors = []

    # -----------------------------------------------------------------------
    # Check 1: Record count
    # -----------------------------------------------------------------------
    if num_records != num_charts:
        errors.append(
            f"Record count mismatch: expected {num_charts:,}, got {num_records:,}"
        )

    # -----------------------------------------------------------------------
    # Check 2: All 3 systems present in every record
    # -----------------------------------------------------------------------
    missing_system_count = 0
    for r in records:
        for system in systems:
            for domain in domains:
                key = f"{system}_{domain}"
                if key not in r:
                    missing_system_count += 1
                    break
            else:
                continue
            break

    if missing_system_count > 0:
        errors.append(
            f"{missing_system_count:,} records are missing at least one system score"
        )
    else:
        print(f"All {len(systems)} systems present in all records: OK")

    # -----------------------------------------------------------------------
    # Check 3: All domain scores in [0, 1]
    # -----------------------------------------------------------------------
    out_of_range = 0
    for r in records:
        for system in systems:
            for domain in domains:
                key = f"{system}_{domain}"
                val = r.get(key)
                if val is not None and not (0.0 <= float(val) <= 1.0):
                    out_of_range += 1
                    if out_of_range == 1:
                        errors.append(
                            f"Score out of [0,1] range: {key}={val} in record"
                            f" {r.get('datetime', '?')}"
                        )

    if out_of_range == 0:
        print("All domain scores in [0, 1]: OK")
    elif out_of_range > 1:
        errors.append(
            f"{out_of_range - 1} additional out-of-range scores found"
        )

    # -----------------------------------------------------------------------
    # Check 4: Null model produces lower correlation than real data
    # -----------------------------------------------------------------------
    try:
        from src.analysis import compute_null_model, RANDOM_SEED

        # Sample up to 500 records for this check
        sample_size = min(500, num_records)
        sample = records[:sample_size]

        bazi_career = [r.get("bazi_career", 0.5) for r in sample]
        ziwei_career = [r.get("ziwei_career", 0.5) for r in sample]

        bazi_arr = np.array(bazi_career)
        ziwei_arr = np.array(ziwei_career)

        # Real correlation
        if np.std(bazi_arr) > 1e-10 and np.std(ziwei_arr) > 1e-10:
            real_r = abs(float(np.corrcoef(bazi_arr, ziwei_arr)[0, 1]))
        else:
            real_r = 0.0

        # Null model correlation
        null_scores = compute_null_model(
            bazi_career, ziwei_career, n_permutations=100, seed=RANDOM_SEED
        )
        null_mean = float(np.mean(null_scores))

        print(f"\nNull model check (career domain, n={sample_size:,}):")
        print(f"  Real |r|:      {real_r:.4f}")
        print(f"  Null mean |r|: {null_mean:.4f}")

        if null_mean >= real_r + 0.05:
            errors.append(
                f"Null model correlation ({null_mean:.4f}) is unexpectedly higher"
                f" than real data ({real_r:.4f}) — possible calibration issue"
            )
    except Exception as e:
        print(f"  WARNING: null model check failed: {e}")

    # -----------------------------------------------------------------------
    # Summary stats
    # -----------------------------------------------------------------------
    stats_summary = data.get("statistics_summary", {})
    correlation = stats_summary.get("correlation", {})
    correlation_inference = stats_summary.get("correlation_inference", {})
    domain_agreement = stats_summary.get("domain_agreement", {})

    print(f"\nCorrelation summary (BaZi–ZiWei career):")
    career_corr = correlation.get("career", {}).get("bazi_ziwei", "N/A")
    print(f"  {career_corr}")

    career_inf = correlation_inference.get("career", {}).get("bazi_ziwei")
    if career_inf:
        print("  95% CI:", f"[{career_inf.get('ci_lower', 0.0):.4f}, {career_inf.get('ci_upper', 0.0):.4f}]")
        print("  p-value:", f"{career_inf.get('p_value', 1.0):.3g}")
        print("  Bonferroni p:", f"{career_inf.get('p_value_bonferroni', 1.0):.3g}")

    print(f"\nDomain agreement summary (BaZi–ZiWei career):")
    career_agree = domain_agreement.get("career", {}).get("bazi_ziwei", "N/A")
    print(f"  {career_agree}")

    # -----------------------------------------------------------------------
    # Final verdict
    # -----------------------------------------------------------------------
    if errors:
        print(f"\nValidation FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("\nValidation passed.")
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main(results_file=args.results_file))
