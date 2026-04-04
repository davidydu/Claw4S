"""Run the full data marketplace experiment with multiprocessing."""

import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

from src.experiment import build_experiment_matrix, run_simulation, ExperimentResult
from src.analysis import aggregate_results, compute_key_findings
from src.report import generate_report

OUTPUT_DIR = "results"


def _run_one(config):
    """Wrapper for multiprocessing (top-level function)."""
    return run_simulation(config)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    configs = build_experiment_matrix()
    n_configs = len(configs)
    n_workers = min(cpu_count(), 8)

    print(f"Running {n_configs} simulations with {n_workers} workers...")
    t0 = time.time()

    results: list[ExperimentResult] = []
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_run_one, configs)):
            results.append(result)
            if (i + 1) % 10 == 0 or (i + 1) == n_configs:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{n_configs}] {elapsed:.1f}s elapsed")

    # Sort by config name for reproducibility
    results.sort(key=lambda r: r.config.name)

    elapsed = time.time() - t0
    print(f"\nAll simulations complete in {elapsed:.1f}s")

    # Analysis
    print("Aggregating results...")
    agg = aggregate_results(results)
    findings = compute_key_findings(agg)

    # Report
    print("Generating report...")
    report = generate_report(results, agg, findings, OUTPUT_DIR)

    # Save raw results as JSON
    raw = []
    for r in results:
        raw.append({
            "config": {
                "composition": r.config.composition,
                "market_size": r.config.market_size,
                "info_regime": r.config.info_regime,
                "seed": r.config.seed,
                "n_rounds": r.config.n_rounds,
            },
            "metrics": r.metrics,
            "audit_scores": r.audit_scores,
            "buyer_welfare": r.buyer_welfare,
            "buyer_surplus": r.buyer_surplus,
            "seller_profit": r.seller_profit,
        })

    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump({
            "metadata": {
                "n_simulations": n_configs,
                "runtime_seconds": elapsed,
            },
            "findings": findings,
            "aggregated": agg,
            "raw_results": raw,
        }, f, indent=2, default=str)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print(f"  results.json  — raw data")
    print(f"  report.md     — summary report")
    print(f"  *.png         — figures")

    # Print key findings
    print("\n=== KEY FINDINGS ===")
    for f in findings:
        print(f"\n{f['finding']}:")
        print(f"  {f['description']}")

    print(f"\nTotal runtime: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
