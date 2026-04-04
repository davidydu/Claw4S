"""Run the full cascading-failures experiment and generate report.

Usage: .venv/bin/python run.py [--diagnostic]

With --diagnostic: runs a small subset (18 sims) for quick validation.
Without flags: runs the full 324-simulation experiment.
"""

import json
import os
import sys

from src.experiment import run_experiment
from src.metrics import aggregate_by_condition
from src.report import generate_report

DIAGNOSTIC = "--diagnostic" in sys.argv


def main() -> None:
    os.makedirs("results", exist_ok=True)

    if DIAGNOSTIC:
        print("[1/3] Running DIAGNOSTIC (18 simulations)...")
        # 1 topology x 1 agent x 3 magnitudes x 2 locations x 3 seeds
        raw_results = run_experiment(
            topology_names=["ring"],
            agent_type_names=["fragile"],
        )
    else:
        print("[1/3] Running full experiment (324 simulations)...")
        raw_results = run_experiment()

    print(f"[2/3] Aggregating metrics...")
    aggregated = aggregate_by_condition(raw_results)

    # Save raw results
    with open("results/results.json", "w") as f:
        json.dump({
            "metadata": {
                "n_simulations": len(raw_results),
                "n_conditions": len(aggregated),
                "diagnostic": DIAGNOSTIC,
                "topologies": sorted({r["topology"] for r in raw_results}),
                "agent_types": sorted({r["agent_type"] for r in raw_results}),
                "shock_magnitudes": sorted({r["shock_magnitude"] for r in raw_results}),
                "shock_locations": sorted({r["shock_location"] for r in raw_results}),
                "seeds": sorted({r["seed"] for r in raw_results}),
                "shock_selection_policy": "hub=max-degree node; random=uniform over non-hub nodes when possible",
            },
            "raw_results": raw_results,
            "aggregated": aggregated,
        }, f, indent=2, default=str)
    print(f"  Saved results/results.json ({len(raw_results)} simulations)")

    print("[3/3] Generating report...")
    report = generate_report(raw_results)
    with open("results/report.md", "w") as f:
        f.write(report)
    print(f"  Saved results/report.md")
    print()
    print(report)


if __name__ == "__main__":
    main()
