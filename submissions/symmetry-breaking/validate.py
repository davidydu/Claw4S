"""Validate symmetry-breaking outputs for executability and scientific sanity.

Must be run from the submissions/symmetry-breaking/ directory.
"""

import json
import os
import sys
from typing import Any, Dict, List, Tuple

RESULTS_DIR = "results"
EXPECTED_HIDDEN_DIMS = [16, 32, 64, 128]
EXPECTED_EPSILONS = [0.0, 1e-6, 1e-4, 1e-2, 1e-1]
MIN_HIGH_EPSILON_BEST_ACC = 0.10
MIN_HIGH_VS_LOW_EPSILON_GAIN = 0.05


def ensure_submission_cwd() -> None:
    """Move to the script directory if called from a different location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.abspath(os.getcwd()) != script_dir:
        print(f"Changing working directory to {script_dir}")
        os.chdir(script_dir)


def check_file_exists(filename: str, errors: List[str]) -> bool:
    """Check that an expected output file exists."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        errors.append(f"Missing file: {path}")
        return False
    return True


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate_results_content(
    results: List[Dict[str, Any]],
    expected_hidden_dims: List[int],
    expected_epsilons: List[float],
) -> Tuple[List[str], Dict[str, float]]:
    """Validate structure and scientific signal in results.json payload."""
    errors: List[str] = []
    diagnostics: Dict[str, float] = {}

    expected_runs = len(expected_hidden_dims) * len(expected_epsilons)
    if len(results) != expected_runs:
        errors.append(f"Expected {expected_runs} runs, got {len(results)}")

    required_fields = [
        "hidden_dim",
        "epsilon",
        "seed",
        "epochs_logged",
        "symmetry_values",
        "loss_values",
        "train_acc_values",
        "final_test_acc",
        "final_train_acc",
        "final_symmetry",
        "initial_symmetry",
        "modulus",
    ]

    for i, run in enumerate(results):
        for field in required_fields:
            if field not in run:
                errors.append(f"Run {i}: missing field '{field}'")

        epochs_logged = run.get("epochs_logged", [])
        symmetry_values = run.get("symmetry_values", [])
        loss_values = run.get("loss_values", [])
        train_acc_values = run.get("train_acc_values", [])

        if epochs_logged and epochs_logged[0] != 0:
            errors.append(f"Run {i}: epochs_logged should start with 0")
        if len(symmetry_values) != len(epochs_logged):
            errors.append(
                f"Run {i}: len(symmetry_values)={len(symmetry_values)} "
                f"!= len(epochs_logged)={len(epochs_logged)}"
            )
        if len(loss_values) != max(0, len(epochs_logged) - 1):
            errors.append(
                f"Run {i}: len(loss_values)={len(loss_values)} should equal "
                f"len(epochs_logged)-1={max(0, len(epochs_logged) - 1)}"
            )
        if len(train_acc_values) != len(loss_values):
            errors.append(
                f"Run {i}: len(train_acc_values)={len(train_acc_values)} "
                f"!= len(loss_values)={len(loss_values)}"
            )

    actual_hidden_dims = sorted(set(r["hidden_dim"] for r in results if "hidden_dim" in r))
    if actual_hidden_dims != sorted(expected_hidden_dims):
        errors.append(
            f"Hidden dims mismatch: expected {sorted(expected_hidden_dims)}, "
            f"got {actual_hidden_dims}"
        )

    actual_epsilons = sorted(set(float(r["epsilon"]) for r in results if "epsilon" in r))
    if actual_epsilons != sorted(expected_epsilons):
        errors.append(
            f"Epsilons mismatch: expected {sorted(expected_epsilons)}, got {actual_epsilons}"
        )

    for run in results:
        init_sym = run.get("initial_symmetry", -1)
        final_sym = run.get("final_symmetry", -1)
        if not (-0.5 <= init_sym <= 1.01):
            errors.append(
                f"hidden={run['hidden_dim']}, eps={run['epsilon']}: "
                f"initial symmetry {init_sym:.4f} out of range [-0.5, 1.01]"
            )
        if not (-0.5 <= final_sym <= 1.01):
            errors.append(
                f"hidden={run['hidden_dim']}, eps={run['epsilon']}: "
                f"final symmetry {final_sym:.4f} out of range [-0.5, 1.01]"
            )

        final_test_acc = run.get("final_test_acc", -1)
        final_train_acc = run.get("final_train_acc", -1)
        if not (0.0 <= final_test_acc <= 1.0):
            errors.append(
                f"hidden={run['hidden_dim']}, eps={run['epsilon']}: "
                f"test accuracy {final_test_acc:.4f} out of [0, 1]"
            )
        if not (0.0 <= final_train_acc <= 1.0):
            errors.append(
                f"hidden={run['hidden_dim']}, eps={run['epsilon']}: "
                f"train accuracy {final_train_acc:.4f} out of [0, 1]"
            )

    zero_eps_runs = [r for r in results if r.get("epsilon") == 0.0]
    for run in zero_eps_runs:
        if run["initial_symmetry"] < 0.99:
            errors.append(
                f"hidden={run['hidden_dim']}, eps=0: "
                f"initial symmetry {run['initial_symmetry']:.4f} should be ~1.0"
            )

    large_eps_runs = [r for r in results if r.get("epsilon", 0.0) >= 0.01]
    broke_count = sum(1 for run in large_eps_runs if run["final_symmetry"] < 0.5)
    diagnostics["broke_count_large_eps"] = float(broke_count)
    diagnostics["num_large_eps_runs"] = float(len(large_eps_runs))

    modulus_values = sorted({int(r["modulus"]) for r in results if "modulus" in r})
    if len(modulus_values) == 1 and modulus_values[0] > 0:
        chance_accuracy = 1.0 / float(modulus_values[0])
        diagnostics["chance_accuracy"] = chance_accuracy
    else:
        errors.append(f"Expected a single positive modulus, got {modulus_values}")
        chance_accuracy = 0.0

    if actual_epsilons:
        min_eps = actual_epsilons[0]
        max_eps = actual_epsilons[-1]
        min_eps_runs = [r for r in results if float(r["epsilon"]) == min_eps]
        max_eps_runs = [r for r in results if float(r["epsilon"]) == max_eps]
        if min_eps_runs and max_eps_runs:
            min_eps_best = max(r["final_test_acc"] for r in min_eps_runs)
            max_eps_best = max(r["final_test_acc"] for r in max_eps_runs)
            gain = max_eps_best - min_eps_best

            diagnostics["best_test_acc_at_min_epsilon"] = float(min_eps_best)
            diagnostics["best_test_acc_at_max_epsilon"] = float(max_eps_best)
            diagnostics["accuracy_gain_max_vs_min_epsilon"] = float(gain)
            diagnostics["min_epsilon"] = float(min_eps)
            diagnostics["max_epsilon"] = float(max_eps)

            min_required_high_eps = max(MIN_HIGH_EPSILON_BEST_ACC, chance_accuracy * 5.0)
            if max_eps_best < min_required_high_eps:
                errors.append(
                    f"Best test accuracy at highest epsilon ({max_eps:.1e}) is "
                    f"{max_eps_best:.4f}, below required {min_required_high_eps:.4f}"
                )
            if gain < MIN_HIGH_VS_LOW_EPSILON_GAIN:
                errors.append(
                    f"Accuracy gain from lowest epsilon ({min_eps:.1e}) to highest epsilon "
                    f"({max_eps:.1e}) is {gain:.4f}, expected >= "
                    f"{MIN_HIGH_VS_LOW_EPSILON_GAIN:.4f}"
                )

    return errors, diagnostics


def validate_summary_content(
    summary: Dict[str, Any],
    results: List[Dict[str, Any]],
    expected_hidden_dims: List[int],
    expected_epsilons: List[float],
    diagnostics: Dict[str, float],
) -> List[str]:
    """Validate summary.json against expected configuration and results."""
    errors: List[str] = []

    if summary.get("num_runs") != len(results):
        errors.append(
            f"summary.num_runs={summary.get('num_runs')} does not match "
            f"results length {len(results)}"
        )

    summary_hidden_dims = sorted(summary.get("hidden_dims", []))
    if summary_hidden_dims != sorted(expected_hidden_dims):
        errors.append(
            f"summary.hidden_dims mismatch: expected {sorted(expected_hidden_dims)}, "
            f"got {summary_hidden_dims}"
        )

    summary_epsilons = sorted(summary.get("epsilons", []))
    if summary_epsilons != sorted(expected_epsilons):
        errors.append(
            f"summary.epsilons mismatch: expected {sorted(expected_epsilons)}, "
            f"got {summary_epsilons}"
        )

    if "chance_accuracy" in summary and "chance_accuracy" in diagnostics:
        if abs(summary["chance_accuracy"] - diagnostics["chance_accuracy"]) > 1e-9:
            errors.append(
                f"summary.chance_accuracy={summary['chance_accuracy']:.10f} "
                f"does not match expected {diagnostics['chance_accuracy']:.10f}"
            )

    zero_eps_runs = [r for r in results if r["epsilon"] == 0.0]
    if zero_eps_runs and "zero_eps_final_symmetry_mean" in summary:
        expected = _mean([r["final_symmetry"] for r in zero_eps_runs])
        if abs(summary["zero_eps_final_symmetry_mean"] - expected) > 1e-8:
            errors.append(
                f"summary.zero_eps_final_symmetry_mean={summary['zero_eps_final_symmetry_mean']:.10f} "
                f"does not match recomputed {expected:.10f}"
            )

    nonzero_eps_runs = [r for r in results if r["epsilon"] > 0.0]
    if nonzero_eps_runs and "nonzero_eps_final_symmetry_mean" in summary:
        expected = _mean([r["final_symmetry"] for r in nonzero_eps_runs])
        if abs(summary["nonzero_eps_final_symmetry_mean"] - expected) > 1e-8:
            errors.append(
                f"summary.nonzero_eps_final_symmetry_mean="
                f"{summary['nonzero_eps_final_symmetry_mean']:.10f} "
                f"does not match recomputed {expected:.10f}"
            )

    for key in [
        "best_test_acc_at_min_epsilon",
        "best_test_acc_at_max_epsilon",
        "accuracy_gain_max_vs_min_epsilon",
    ]:
        if key in summary and key in diagnostics:
            if abs(summary[key] - diagnostics[key]) > 1e-8:
                errors.append(
                    f"summary.{key}={summary[key]:.10f} does not match "
                    f"recomputed {diagnostics[key]:.10f}"
                )

    return errors


def main() -> int:
    """Entry point for CLI validation."""
    ensure_submission_cwd()

    errors: List[str] = []

    print("Checking output files...")
    expected_files = [
        "results.json",
        "summary.json",
        "report.md",
        "symmetry_trajectories.png",
        "accuracy_vs_epsilon.png",
        "symmetry_heatmap.png",
    ]
    for filename in expected_files:
        exists = check_file_exists(filename, errors)
        print(f"  {'OK' if exists else 'MISSING':>7}  {filename}")

    results: List[Dict[str, Any]] = []
    diagnostics: Dict[str, float] = {}

    print("\nValidating results.json...")
    results_path = os.path.join(RESULTS_DIR, "results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        print(
            f"  Number of runs: {len(results)} "
            f"(expected {len(EXPECTED_HIDDEN_DIMS) * len(EXPECTED_EPSILONS)})"
        )
        results_errors, diagnostics = validate_results_content(
            results=results,
            expected_hidden_dims=EXPECTED_HIDDEN_DIMS,
            expected_epsilons=EXPECTED_EPSILONS,
        )
        errors.extend(results_errors)

        broke_count = int(diagnostics.get("broke_count_large_eps", 0))
        large_eps_total = int(diagnostics.get("num_large_eps_runs", 0))
        print(
            f"  Runs with eps>=0.01 that broke symmetry (<0.5): "
            f"{broke_count}/{large_eps_total}"
        )
        if "chance_accuracy" in diagnostics:
            print(f"  Chance-level accuracy: {diagnostics['chance_accuracy']:.4f}")
        if "best_test_acc_at_max_epsilon" in diagnostics:
            print(
                f"  Best test accuracy at highest epsilon: "
                f"{diagnostics['best_test_acc_at_max_epsilon']:.4f}"
            )
        if "accuracy_gain_max_vs_min_epsilon" in diagnostics:
            print(
                f"  Best-accuracy gain (highest vs lowest epsilon): "
                f"{diagnostics['accuracy_gain_max_vs_min_epsilon']:.4f}"
            )

    print("\nValidating summary.json...")
    summary_path = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

        print(f"  Hidden dims: {summary.get('hidden_dims', 'MISSING')}")
        print(f"  Epsilons: {summary.get('epsilons', 'MISSING')}")

        if "zero_eps_final_symmetry_mean" in summary:
            print(
                f"  Zero-eps mean final symmetry: "
                f"{summary['zero_eps_final_symmetry_mean']:.4f}"
            )
        if "nonzero_eps_final_symmetry_mean" in summary:
            print(
                f"  Non-zero-eps mean final symmetry: "
                f"{summary['nonzero_eps_final_symmetry_mean']:.4f}"
            )
        if "median_breaking_epoch" in summary:
            print(
                f"  Median breaking epoch (eps>=1e-4): "
                f"{summary['median_breaking_epoch']:.0f}"
            )
        if "best_test_acc" in summary:
            print(f"  Best run test accuracy: {summary['best_test_acc']:.4f}")
        if "accuracy_gain_max_vs_min_epsilon" in summary:
            print(
                f"  Summary gain (max eps vs min eps): "
                f"{summary['accuracy_gain_max_vs_min_epsilon']:.4f}"
            )

        if results:
            errors.extend(
                validate_summary_content(
                    summary=summary,
                    results=results,
                    expected_hidden_dims=EXPECTED_HIDDEN_DIMS,
                    expected_epsilons=EXPECTED_EPSILONS,
                    diagnostics=diagnostics,
                )
            )

    print()
    if errors:
        print(f"Validation FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"  - {err}")
        return 1

    print("Validation passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
