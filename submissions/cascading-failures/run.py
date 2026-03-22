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
        # Override experiment to run only 1 topology x 1 agent x 3 mags x 2 locs x 3 seeds
        from src import experiment as exp
        orig_topos = exp.TOPOLOGY_NAMES
        orig_agents = exp.AGENT_TYPE_NAMES
        exp.TOPOLOGY_NAMES = ["ring"]
        exp.AGENT_TYPE_NAMES = ["fragile"]
        try:
            raw_results = run_experiment()
        finally:
            exp.TOPOLOGY_NAMES = orig_topos
            exp.AGENT_TYPE_NAMES = orig_agents
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
