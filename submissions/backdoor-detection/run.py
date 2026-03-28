"""Run the full backdoor detection experiment sweep and generate report.

Must be run from the submission directory: submissions/backdoor-detection/
"""

import json
import os
import sys
import time
from dataclasses import asdict

from src.cli import ensure_submission_cwd
from src.experiment import run_sweep
from src.report import generate_report, generate_figures


def main() -> None:
    """Execute the full experiment pipeline."""
    try:
        ensure_submission_cwd(__file__)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print("[1/4] Running parameter sweep (36 experiments)...")
    start = time.time()

    def progress(i: int, total: int, result) -> None:
        pf = result.config["poison_fraction"]
        ts = result.config["trigger_strength"]
        hd = result.config["hidden_dim"]
        auc = result.detection_auc
        print(f"  [{i}/{total}] poison={pf*100:.0f}% strength={ts} "
              f"hidden={hd} -> AUC={auc:.3f} ({result.elapsed_seconds:.1f}s)")

    results = run_sweep(
        poison_fractions=[0.05, 0.10, 0.20, 0.30],
        trigger_strengths=[3.0, 5.0, 10.0],
        hidden_dims=[64, 128, 256],
        seed=42,
        progress_callback=progress,
    )

    elapsed = time.time() - start
    print(f"  Sweep completed in {elapsed:.1f}s")

    # Convert results to dicts for serialization
    results_dicts = [asdict(r) for r in results]

    print("[2/4] Saving results to results/results.json")
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump({
            "metadata": {
                "n_experiments": len(results_dicts),
                "poison_fractions": [0.05, 0.10, 0.20, 0.30],
                "trigger_strengths": [3.0, 5.0, 10.0],
                "hidden_dims": [64, 128, 256],
                "seed": 42,
                "total_elapsed_seconds": elapsed,
            },
            "results": results_dicts,
        }, f, indent=2)

    print("[3/4] Generating report...")
    report = generate_report(results_dicts, output_dir)
    print(report)

    print("[4/4] Generating figures...")
    figure_paths = generate_figures(results_dicts, output_dir)
    for p in figure_paths:
        print(f"  Saved: {p}")

    print(f"\nDone. {len(results_dicts)} experiments saved to {output_dir}/")


if __name__ == "__main__":
    main()
