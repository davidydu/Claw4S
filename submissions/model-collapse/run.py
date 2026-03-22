"""Run the full model-collapse experiment and save results.

Executes 135 simulations (3 agent types x 5 GT fractions x 3 distributions
x 3 seeds, 10 generations each) in parallel, then generates a report.
"""

import json
import os
import time

from src.simulation import build_configs, run_experiment
from src.analysis import aggregate_by_condition, build_summary
from src.report import generate_report

RESULTS_DIR = "results"


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/3] Building experiment grid...")
    configs = build_configs()
    print(f"      {len(configs)} simulations queued")

    print("[2/3] Running simulations (multiprocessing)...")
    t0 = time.time()
    results = run_experiment(configs)
    elapsed = time.time() - t0
    print(f"      Completed in {elapsed:.1f}s")

    # Save raw results
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "n_simulations": len(results),
                "agent_types": sorted({r["config"]["agent_type"] for r in results}),
                "gt_fractions": sorted({r["config"]["gt_fraction"] for r in results}),
                "distributions": sorted({r["config"]["dist_name"] for r in results}),
                "n_generations": results[0]["config"]["n_generations"],
                "elapsed_seconds": round(elapsed, 1),
            },
            "results": results,
        }, f, indent=2)
    print(f"      Raw results saved to {results_path}")

    # Save summary
    aggregated = aggregate_by_condition(results)
    summary = build_summary(aggregated)
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      Summary saved to {summary_path}")

    print("[3/3] Generating report...")
    report = generate_report(results)
    report_path = os.path.join(RESULTS_DIR, "report.md")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"      Report saved to {report_path}")

    print()
    print("Done. Key files:")
    print(f"  {results_path}")
    print(f"  {summary_path}")
    print(f"  {report_path}")


if __name__ == "__main__":
    main()
