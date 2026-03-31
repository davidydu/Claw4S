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

POISON_FRACTIONS = [0.05, 0.10, 0.20, 0.30]
TRIGGER_STRENGTHS = [3.0, 5.0, 10.0]
HIDDEN_DIMS = [64, 128, 256]
SEED = 42


def build_results_payload(
    results_dicts: list[dict],
    poison_fractions: list[float],
    trigger_strengths: list[float],
    hidden_dims: list[int],
    seed: int,
) -> dict:
    """Build a deterministic JSON payload for results/results.json."""
    stable_results = []
    for result in results_dicts:
        stable_result = dict(result)
        stable_result.pop("elapsed_seconds", None)
        stable_results.append(stable_result)

    return {
        "metadata": {
            "n_experiments": len(stable_results),
            "poison_fractions": poison_fractions,
            "trigger_strengths": trigger_strengths,
            "hidden_dims": hidden_dims,
            "seed": seed,
        },
        "results": stable_results,
    }


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
        poison_fractions=POISON_FRACTIONS,
        trigger_strengths=TRIGGER_STRENGTHS,
        hidden_dims=HIDDEN_DIMS,
        seed=SEED,
        progress_callback=progress,
    )

    elapsed = time.time() - start
    print(f"  Sweep completed in {elapsed:.1f}s")

    # Convert results to dicts for serialization
    results_dicts = [asdict(r) for r in results]
    results_payload = build_results_payload(
        results_dicts=results_dicts,
        poison_fractions=POISON_FRACTIONS,
        trigger_strengths=TRIGGER_STRENGTHS,
        hidden_dims=HIDDEN_DIMS,
        seed=SEED,
    )

    print("[2/4] Saving results to results/results.json")
    with open(os.path.join(output_dir, "results.json"), "w") as f:
        json.dump(results_payload, f, indent=2)

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
