"""Validate symmetry-breaking experiment results for completeness and correctness.

Must be run from the submissions/symmetry-breaking/ directory.
"""

import json
import os
import sys

# Working-directory guard
script_dir = os.path.dirname(os.path.abspath(__file__))
if os.path.abspath(os.getcwd()) != script_dir:
    print(f"Changing working directory to {script_dir}")
    os.chdir(script_dir)

RESULTS_DIR = "results"
errors = []


def check_file_exists(filename: str) -> bool:
    """Check that an expected output file exists."""
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        errors.append(f"Missing file: {path}")
        return False
    return True


# 1. Check all expected files exist
print("Checking output files...")
expected_files = [
    "results.json",
    "summary.json",
    "report.md",
    "symmetry_trajectories.png",
    "accuracy_vs_epsilon.png",
    "symmetry_heatmap.png",
]
for f in expected_files:
    exists = check_file_exists(f)
    print(f"  {'OK' if exists else 'MISSING':>7}  {f}")

# 2. Validate results.json structure and content
print("\nValidating results.json...")
if os.path.exists(os.path.join(RESULTS_DIR, "results.json")):
    with open(os.path.join(RESULTS_DIR, "results.json")) as f:
        results = json.load(f)

    num_runs = len(results)
    expected_runs = 20  # 4 hidden_dims x 5 epsilons
    print(f"  Number of runs: {num_runs} (expected {expected_runs})")
    if num_runs != expected_runs:
        errors.append(f"Expected {expected_runs} runs, got {num_runs}")

    # Check each run has required fields
    required_fields = [
        "hidden_dim", "epsilon", "seed", "epochs_logged",
        "symmetry_values", "final_test_acc", "final_symmetry",
        "initial_symmetry",
    ]
    for i, r in enumerate(results):
        for field in required_fields:
            if field not in r:
                errors.append(f"Run {i}: missing field '{field}'")

    # Check symmetry metric ranges
    for r in results:
        init_sym = r.get("initial_symmetry", -1)
        final_sym = r.get("final_symmetry", -1)
        if not (-0.5 <= init_sym <= 1.01):
            errors.append(
                f"hidden={r['hidden_dim']}, eps={r['epsilon']}: "
                f"initial symmetry {init_sym:.4f} out of range [-0.5, 1.01]"
            )
        if not (-0.5 <= final_sym <= 1.01):
            errors.append(
                f"hidden={r['hidden_dim']}, eps={r['epsilon']}: "
                f"final symmetry {final_sym:.4f} out of range [-0.5, 1.01]"
            )

    # Key scientific validation: symmetric init (eps=0) should start near 1.0
    zero_eps_runs = [r for r in results if r["epsilon"] == 0.0]
    for r in zero_eps_runs:
        if r["initial_symmetry"] < 0.99:
            errors.append(
                f"hidden={r['hidden_dim']}, eps=0: "
                f"initial symmetry {r['initial_symmetry']:.4f} should be ~1.0"
            )

    # Perturbed init (eps=1e-1) should eventually break symmetry
    large_eps_runs = [r for r in results if r["epsilon"] >= 0.01]
    broke_count = sum(1 for r in large_eps_runs if r["final_symmetry"] < 0.5)
    print(f"  Runs with eps>=0.01 that broke symmetry (<0.5): {broke_count}/{len(large_eps_runs)}")

    # Check that accuracy values are in [0, 1]
    for r in results:
        if not (0.0 <= r["final_test_acc"] <= 1.0):
            errors.append(
                f"hidden={r['hidden_dim']}, eps={r['epsilon']}: "
                f"test accuracy {r['final_test_acc']:.4f} out of [0, 1]"
            )

# 3. Validate summary.json
print("\nValidating summary.json...")
if os.path.exists(os.path.join(RESULTS_DIR, "summary.json")):
    with open(os.path.join(RESULTS_DIR, "summary.json")) as f:
        summary = json.load(f)

    print(f"  Hidden dims: {summary.get('hidden_dims', 'MISSING')}")
    print(f"  Epsilons: {summary.get('epsilons', 'MISSING')}")

    if "zero_eps_final_symmetry_mean" in summary:
        print(f"  Zero-eps mean final symmetry: {summary['zero_eps_final_symmetry_mean']:.4f}")
    if "nonzero_eps_final_symmetry_mean" in summary:
        print(f"  Non-zero-eps mean final symmetry: {summary['nonzero_eps_final_symmetry_mean']:.4f}")
    if "median_breaking_epoch" in summary:
        print(f"  Median breaking epoch (eps>=1e-4): {summary['median_breaking_epoch']:.0f}")

# 4. Final verdict
print()
if errors:
    print(f"Validation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation passed.")
