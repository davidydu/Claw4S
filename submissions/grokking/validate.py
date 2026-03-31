"""Validate grokking phase diagram outputs for completeness and consistency."""

import argparse
import json
import math
import os
import sys
from collections import Counter
from itertools import product

from src.analysis import aggregate_results, classify_phase, compute_grokking_gap
from src.train import TrainResult


def check(condition: bool, message: str, errors: list[str]) -> None:
    """Record an error if condition is False."""
    if not condition:
        errors.append(message)


def build_required_files(
    hidden_dims: list[int], results_dir: str = "results"
) -> list[str]:
    """Build expected artifact list from hidden dims."""
    files = [
        os.path.join(results_dir, "sweep_results.json"),
        os.path.join(results_dir, "phase_diagram.json"),
        os.path.join(results_dir, "metadata.json"),
        os.path.join(results_dir, "report.md"),
        os.path.join(results_dir, "grokking_curves.png"),
    ]
    for hd in sorted(set(hidden_dims)):
        files.append(os.path.join(results_dir, f"phase_diagram_h{hd}.png"))
    return files


def find_grid_coverage_errors(sweep: list[dict]) -> list[str]:
    """Ensure every Cartesian grid point appears exactly once."""
    if not sweep:
        return ["Sweep is empty; cannot validate grid coverage."]

    errors = []
    weight_decays = sorted(set(r["config"]["weight_decay"] for r in sweep))
    fractions = sorted(set(r["config"]["train_fraction"] for r in sweep))
    hidden_dims = sorted(set(r["config"]["hidden_dim"] for r in sweep))

    combo_counts = Counter(
        (
            r["config"]["hidden_dim"],
            r["config"]["weight_decay"],
            r["config"]["train_fraction"],
        )
        for r in sweep
    )

    expected = set(product(hidden_dims, weight_decays, fractions))
    for combo in sorted(expected):
        count = combo_counts.get(combo, 0)
        if count == 0:
            errors.append(
                "Missing grid point: "
                f"hidden_dim={combo[0]}, weight_decay={combo[1]}, "
                f"train_fraction={combo[2]}"
            )
        elif count > 1:
            errors.append(
                "Duplicate grid point: "
                f"hidden_dim={combo[0]}, weight_decay={combo[1]}, "
                f"train_fraction={combo[2]} appears {count} times"
            )
    return errors


def _expected_phase_and_gap(run_result: dict) -> tuple[str, int | None]:
    """Recompute phase classification from logged metrics."""
    metrics = run_result["metrics"]
    reconstructed = TrainResult(
        final_train_acc=metrics["final_train_acc"],
        final_test_acc=metrics["final_test_acc"],
        epoch_train_95=metrics.get("epoch_train_95"),
        epoch_test_95=metrics.get("epoch_test_95"),
    )
    expected_phase = classify_phase(reconstructed).value
    expected_gap = compute_grokking_gap(reconstructed)
    return expected_phase, expected_gap


def find_phase_consistency_errors(sweep: list[dict]) -> list[str]:
    """Validate per-run phase and metric consistency."""
    errors = []
    for idx, run_result in enumerate(sweep):
        metrics = run_result["metrics"]
        config = run_result["config"]
        run_label = (
            f"run[{idx}] (h={config.get('hidden_dim')}, "
            f"wd={config.get('weight_decay')}, "
            f"frac={config.get('train_fraction')})"
        )

        expected_phase, expected_gap = _expected_phase_and_gap(run_result)
        if run_result["phase"] != expected_phase:
            errors.append(
                f"Phase mismatch for {run_label}: "
                f"recorded={run_result['phase']}, expected={expected_phase}"
            )

        recorded_gap = run_result.get("grokking_gap")
        if recorded_gap != expected_gap:
            errors.append(
                f"Grokking gap mismatch for {run_label}: "
                f"recorded={recorded_gap}, expected={expected_gap}"
            )

        history_lengths = [
            len(metrics["train_accs"]),
            len(metrics["test_accs"]),
            len(metrics["train_losses"]),
            len(metrics["test_losses"]),
            len(metrics["logged_epochs"]),
        ]
        if len(set(history_lengths)) != 1:
            errors.append(f"Metric history length mismatch for {run_label}")

        logged_epochs = metrics["logged_epochs"]
        if logged_epochs != sorted(logged_epochs):
            errors.append(f"Logged epochs not sorted for {run_label}")
        if logged_epochs and logged_epochs[-1] > metrics["total_epochs"]:
            errors.append(
                f"Logged epoch exceeds total_epochs for {run_label}: "
                f"{logged_epochs[-1]} > {metrics['total_epochs']}"
            )
    return errors


def find_phase_summary_errors(sweep: list[dict], summary: dict) -> list[str]:
    """Validate summary statistics against recomputed values."""
    aggregate_input = [
        {"phase": r["phase"], "grokking_gap": r.get("grokking_gap")}
        for r in sweep
    ]
    expected = aggregate_results(aggregate_input)

    errors = []
    if summary.get("phase_counts") != expected["phase_counts"]:
        errors.append(
            "phase_counts mismatch: "
            f"recorded={summary.get('phase_counts')}, "
            f"expected={expected['phase_counts']}"
        )

    if summary.get("total_runs") != expected["total_runs"]:
        errors.append(
            "total_runs mismatch: "
            f"recorded={summary.get('total_runs')}, "
            f"expected={expected['total_runs']}"
        )

    if not math.isclose(
        float(summary.get("grokking_fraction", -1)),
        float(expected["grokking_fraction"]),
        rel_tol=1e-9,
        abs_tol=1e-9,
    ):
        errors.append(
            "grokking_fraction mismatch: "
            f"recorded={summary.get('grokking_fraction')}, "
            f"expected={expected['grokking_fraction']}"
        )

    for key in ["mean_grokking_gap", "max_grokking_gap"]:
        if summary.get(key) != expected[key]:
            errors.append(
                f"{key} mismatch: recorded={summary.get(key)}, expected={expected[key]}"
            )

    return errors


def find_metadata_errors(sweep: list[dict], metadata: dict) -> list[str]:
    """Cross-check metadata against observed sweep contents."""
    errors = []
    expected_total = metadata.get("expected_total_runs")
    if expected_total != len(sweep):
        errors.append(
            "metadata expected_total_runs mismatch: "
            f"recorded={expected_total}, observed={len(sweep)}"
        )

    sweep_cfg = metadata.get("sweep", {})
    observed_wd = sorted(set(r["config"]["weight_decay"] for r in sweep))
    observed_frac = sorted(set(r["config"]["train_fraction"] for r in sweep))
    observed_hd = sorted(set(r["config"]["hidden_dim"] for r in sweep))
    observed_seeds = sorted(set(r["config"].get("seed") for r in sweep))

    if sweep_cfg.get("weight_decays") != observed_wd:
        errors.append("metadata sweep.weight_decays does not match observed grid")
    if sweep_cfg.get("dataset_fractions") != observed_frac:
        errors.append("metadata sweep.dataset_fractions does not match observed grid")
    if sweep_cfg.get("hidden_dims") != observed_hd:
        errors.append("metadata sweep.hidden_dims does not match observed grid")
    if len(observed_seeds) == 1 and sweep_cfg.get("seed") != observed_seeds[0]:
        errors.append("metadata sweep.seed does not match observed run seeds")
    if len(observed_seeds) > 1:
        errors.append(f"Multiple seeds detected in sweep results: {observed_seeds}")

    return errors


def _load_json(path: str, errors: list[str]) -> dict | list | None:
    """Load JSON file with consistent error reporting."""
    try:
        with open(path, encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except FileNotFoundError:
        errors.append(f"Could not open {path}")
    except json.JSONDecodeError as exc:
        errors.append(f"Invalid JSON in {path}: {exc}")
    return None


def validate_results(results_dir: str = "results") -> list[str]:
    """Run all validation checks and return a list of errors."""
    errors: list[str] = []

    sweep_path = os.path.join(results_dir, "sweep_results.json")
    sweep = _load_json(sweep_path, errors)
    if not isinstance(sweep, list):
        sweep = []

    hidden_dims = (
        sorted(set(r["config"]["hidden_dim"] for r in sweep))
        if sweep
        else [16, 32, 64]
    )
    required_files = build_required_files(hidden_dims, results_dir)

    print("Checking result files...")
    for file_path in required_files:
        exists = os.path.isfile(file_path)
        check(exists, f"Missing file: {file_path}", errors)
        print(f"  {'OK' if exists else 'MISSING'}: {file_path}")

    print("\nValidating sweep results...")
    if sweep:
        n_runs = len(sweep)
        print(f"  Total runs: {n_runs}")

        valid_phases = {"confusion", "memorization", "grokking", "comprehension"}
        for idx, run_result in enumerate(sweep):
            check("config" in run_result, f"Missing config in run[{idx}]", errors)
            check("metrics" in run_result, f"Missing metrics in run[{idx}]", errors)
            check("phase" in run_result, f"Missing phase in run[{idx}]", errors)
            if "phase" in run_result:
                check(
                    run_result["phase"] in valid_phases,
                    f"Invalid phase in run[{idx}]: {run_result['phase']}",
                    errors,
                )

            if "metrics" in run_result:
                metrics = run_result["metrics"]
                for key in [
                    "final_train_acc",
                    "final_test_acc",
                    "train_accs",
                    "test_accs",
                    "train_losses",
                    "test_losses",
                    "logged_epochs",
                    "total_epochs",
                ]:
                    check(
                        key in metrics,
                        f"Missing metrics.{key} in run[{idx}]",
                        errors,
                    )
                if "final_train_acc" in metrics:
                    check(
                        0.0 <= metrics["final_train_acc"] <= 1.0,
                        f"Train acc out of range in run[{idx}]: "
                        f"{metrics['final_train_acc']}",
                        errors,
                    )
                if "final_test_acc" in metrics:
                    check(
                        0.0 <= metrics["final_test_acc"] <= 1.0,
                        f"Test acc out of range in run[{idx}]: "
                        f"{metrics['final_test_acc']}",
                        errors,
                    )

            if "config" in run_result:
                config = run_result["config"]
                check(
                    config.get("param_count", 0) < 100_000,
                    f"Param count exceeds 100K in run[{idx}]: "
                    f"{config.get('param_count')}",
                    errors,
                )

        phases = [r["phase"] for r in sweep if "phase" in r]
        phase_counts = Counter(phases)
        print(f"  Phase distribution: {dict(phase_counts)}")

        weight_decays = sorted(set(r["config"]["weight_decay"] for r in sweep))
        fractions = sorted(set(r["config"]["train_fraction"] for r in sweep))
        hidden_dims = sorted(set(r["config"]["hidden_dim"] for r in sweep))
        print(f"  Weight decays: {weight_decays}")
        print(f"  Dataset fractions: {fractions}")
        print(f"  Hidden dims: {hidden_dims}")

        errors.extend(find_grid_coverage_errors(sweep))
        errors.extend(find_phase_consistency_errors(sweep))
    else:
        errors.append("Sweep results missing or malformed.")

    print("\nValidating phase summary...")
    summary_path = os.path.join(results_dir, "phase_diagram.json")
    summary = _load_json(summary_path, errors)
    if isinstance(summary, dict):
        for key in ["phase_counts", "total_runs", "grokking_fraction"]:
            check(key in summary, f"Missing '{key}' in phase summary", errors)
        if sweep:
            errors.extend(find_phase_summary_errors(sweep, summary))
        print(f"  Grokking fraction: {summary.get('grokking_fraction', 0):.1%}")
    else:
        errors.append("Phase summary missing or malformed.")

    print("\nValidating metadata...")
    metadata_path = os.path.join(results_dir, "metadata.json")
    metadata = _load_json(metadata_path, errors)
    if isinstance(metadata, dict):
        for key in ["expected_total_runs", "sweep", "environment"]:
            check(key in metadata, f"Missing '{key}' in metadata", errors)
        if sweep:
            errors.extend(find_metadata_errors(sweep, metadata))
        print(f"  Metadata schema version: {metadata.get('schema_version')}")
    else:
        errors.append("Metadata missing or malformed.")

    print("\nChecking file sizes...")
    for file_path in required_files:
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            check(size > 0, f"File is empty: {file_path}", errors)
            print(f"  {file_path}: {size:,} bytes")

    return errors


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Validate grokking outputs.")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing run artifacts (default: results).",
    )
    args = parser.parse_args()

    errors = validate_results(results_dir=args.results_dir)
    print()
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)

    print("Validation passed.")


if __name__ == "__main__":
    main()
