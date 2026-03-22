"""Run the full information cascade experiment and generate report.

Executes 216 simulations (4 agent types x 3 signal qualities x 3 sequence
lengths x 2 true states x 3 seeds) using multiprocessing, computes aggregate
metrics, and saves results to results/.
"""

import json
import os
import time

from src.experiment import (
    AGENT_TYPES,
    SEQUENCE_LENGTHS,
    SIGNAL_QUALITIES,
    run_experiment,
    compute_all_metrics,
)
from src.report import generate_report

RESULTS_DIR = "results"


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("[1/4] Running 216 cascade simulations...")
    t0 = time.time()
    raw_results = run_experiment()
    elapsed = time.time() - t0
    print(f"      Completed {len(raw_results)} simulations in {elapsed:.1f}s")

    print("[2/4] Computing aggregate metrics...")
    metrics = compute_all_metrics(raw_results)

    print("[3/4] Generating report...")
    metadata = {
        "n_simulations": len(raw_results),
        "runtime_s": elapsed,
        "n_agent_types": len(AGENT_TYPES),
        "signal_qualities": SIGNAL_QUALITIES,
        "sequence_lengths": SEQUENCE_LENGTHS,
    }
    report = generate_report(metrics, metadata)

    print(f"[4/4] Saving results to {RESULTS_DIR}/")
    # Save raw results (actions/signals as lists for JSON)
    with open(os.path.join(RESULTS_DIR, "raw_results.json"), "w") as f:
        json.dump(raw_results, f, indent=2)

    # Save aggregate metrics
    with open(os.path.join(RESULTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Save metadata
    with open(os.path.join(RESULTS_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save report
    with open(os.path.join(RESULTS_DIR, "report.md"), "w") as f:
        f.write(report)

    print(report)
    print(f"\nDone. Results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
