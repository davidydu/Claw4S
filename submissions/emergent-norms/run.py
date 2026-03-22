"""Run the full emergent norms experiment and generate report.

Usage: .venv/bin/python run.py
"""

import json
import os
import sys
import time

from src.experiment import run_experiment
from src.report import generate_report


def main() -> None:
    os.makedirs("results", exist_ok=True)

    print("[1/3] Running experiment (108 simulations, 50k rounds each)...")
    t0 = time.perf_counter()
    results = run_experiment()
    elapsed = time.perf_counter() - t0
    print(f"      Completed in {elapsed:.1f}s")

    print("[2/3] Generating report...")
    report = generate_report(results)

    print("[3/3] Saving results to results/")
    with open("results/results.json", "w") as f:
        json.dump({"metadata": _metadata(results, elapsed), "results": results}, f, indent=2)

    with open("results/report.md", "w") as f:
        f.write(report)

    print(report)
    print("Done. Output saved to results/results.json and results/report.md")


def _metadata(results: list[dict], elapsed: float) -> dict:
    return {
        "num_simulations": len(results),
        "total_rounds_per_sim": results[0]["total_rounds"],
        "population_sizes": sorted(set(r["population_size"] for r in results)),
        "games": sorted(set(r["game"] for r in results)),
        "compositions": sorted(set(r["composition_name"] for r in results)),
        "elapsed_seconds": round(elapsed, 1),
    }


if __name__ == "__main__":
    main()
