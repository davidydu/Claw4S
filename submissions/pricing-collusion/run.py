# run.py
"""Run the full pricing collusion experiment and generate report."""

import json
import os
import numpy as np

from src.experiment import ExperimentConfig, run_simulation, MATCHUPS
from src.analysis import analyze_results
from src.report import generate_report, generate_figures

MEMORIES = [1, 3, 5]
PRESETS = ["e-commerce", "ride-share", "commodity"]
SEEDS = list(range(5))
SHOCK_CONDITIONS = [False, True]


def build_configs(total_rounds=500_000):
    """Build the full experiment matrix."""
    configs = []
    for matchup in MATCHUPS:
        for memory in MEMORIES:
            for preset in PRESETS:
                for shocks in SHOCK_CONDITIONS:
                    for seed in SEEDS:
                        configs.append(ExperimentConfig(
                            matchup=matchup, memory=memory, preset=preset,
                            shocks=shocks, seed=seed,
                            total_rounds=total_rounds,
                        ))
    return configs


def main():
    os.makedirs("results", exist_ok=True)

    configs = build_configs()
    total = len(configs)
    print(f"[1/5] Running {total} simulations...")

    results = []
    for i, config in enumerate(configs):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Simulation {i + 1}/{total}: "
                  f"{config.matchup}/M{config.memory}/{config.preset}"
                  f"/{'shock' if config.shocks else 'no-shock'}/seed{config.seed}")
        results.append(run_simulation(config))

    print(f"[2/5] Running auditor panel on {total} results...")
    analysis = analyze_results(results)

    print("[3/5] Computing statistics...")
    # Already done in analyze_results

    print("[4/5] Generating report and figures...")
    report = generate_report(analysis)
    generate_figures(analysis)

    print("[5/5] Saving results to results/")
    # Save results JSON (without numpy arrays — just summary data)
    serializable = {
        "metadata": {
            "num_simulations": total,
            "num_conditions": len(analysis["statistics"]),
            "matchups": list(MATCHUPS.keys()),
            "memories": MEMORIES,
            "presets": PRESETS,
            "seeds": SEEDS,
        },
        "records": [
            {k: v for k, v in r.items()
             if k not in ("auditor_evidence",)}  # skip large evidence dicts
            for r in analysis["records"]
        ],
        "statistics": analysis["statistics"],
    }

    with open("results/results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    with open("results/report.md", "w") as f:
        f.write(report)

    with open("results/statistical_tests.json", "w") as f:
        stat_data = [
            {k: v for k, v in s.items()}
            for s in analysis["statistics"]
        ]
        json.dump(stat_data, f, indent=2)

    print(f"\nDone. Results saved to results/")
    print(f"  results/results.json ({total} simulation records)")
    print(f"  results/report.md")
    print(f"  results/statistical_tests.json")


if __name__ == "__main__":
    main()
