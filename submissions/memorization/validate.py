"""Validate memorization capacity results for completeness and correctness."""

import json
import os
import sys

# Ensure we are running from the submission directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

RESULTS_PATH = "results/results.json"

if not os.path.exists(RESULTS_PATH):
    print(f"ERROR: {RESULTS_PATH} not found. Run run.py first.")
    sys.exit(1)

with open(RESULTS_PATH) as f:
    data = json.load(f)

errors = []

# Check reproducibility metadata
run_metadata = data.get("run_metadata")
if not isinstance(run_metadata, dict):
    errors.append(
        "Missing run_metadata in results.json. Re-run run.py to regenerate reproducibility metadata."
    )
else:
    required_fields = [
        "seeds",
        "hidden_dims",
        "n_train",
        "n_test",
        "d",
        "n_classes",
        "max_epochs",
        "lr",
        "python_version",
        "dependency_versions",
        "execution",
    ]
    missing = [key for key in required_fields if key not in run_metadata]
    if missing:
        errors.append(f"run_metadata missing required fields: {missing}")
    else:
        print(f"Run metadata: Python {run_metadata['python_version']}")
        print(f"  Seeds: {run_metadata['seeds']}")
        print(f"  Hidden dims: {run_metadata['hidden_dims']}")

        dep_versions = run_metadata.get("dependency_versions", {})
        for dep in ["torch", "numpy", "scipy"]:
            if dep not in dep_versions:
                errors.append(f"run_metadata.dependency_versions missing '{dep}'")

        execution = run_metadata.get("execution", {})
        for key in ["start_utc", "end_utc", "plots_generated"]:
            if key not in execution:
                errors.append(f"run_metadata.execution missing '{key}'")

        if isinstance(run_metadata.get("seeds"), list) and len(run_metadata["seeds"]) < 2:
            errors.append(
                f"Expected >= 2 seeds in run_metadata for variance estimation, got {run_metadata['seeds']}"
            )

# Check metadata
metadata = data.get("metadata", {})
n_train = metadata.get("n_train", 0)
n_classes = metadata.get("n_classes", 0)
hidden_dims = metadata.get("hidden_dims", [])
label_types = metadata.get("label_types", [])

print(f"Training samples: {n_train}")
print(f"Number of classes: {n_classes}")
print(f"Hidden dims swept: {hidden_dims}")
print(f"Label types: {label_types}")

if n_train < 100:
    errors.append(f"Expected n_train >= 100, got {n_train}")
if n_classes < 2:
    errors.append(f"Expected n_classes >= 2, got {n_classes}")
if len(hidden_dims) < 4:
    errors.append(f"Expected >= 4 hidden dims, got {len(hidden_dims)}")
if len(label_types) < 2:
    errors.append(f"Expected >= 2 label types, got {len(label_types)}")

if isinstance(run_metadata, dict):
    for key in ["n_train", "n_test", "d", "n_classes", "max_epochs"]:
        if key in run_metadata and metadata.get(key) != run_metadata.get(key):
            errors.append(
                f"Mismatch between metadata.{key}={metadata.get(key)} and "
                f"run_metadata.{key}={run_metadata.get(key)}"
            )

    if "hidden_dims" in run_metadata and run_metadata["hidden_dims"] != hidden_dims:
        errors.append(
            "Mismatch between metadata.hidden_dims and run_metadata.hidden_dims "
            f"({hidden_dims} vs {run_metadata['hidden_dims']})"
        )

# Check results
results = data.get("results", [])
expected_runs = len(hidden_dims) * len(label_types)
print(f"Total runs: {len(results)} (expected {expected_runs})")

if len(results) != expected_runs:
    errors.append(f"Expected {expected_runs} runs, got {len(results)}")

# Check that random labels achieve high train accuracy for large models
random_results = [r for r in results if r["label_type"] == "random"]
if random_results:
    max_random_acc = max(r["train_acc"] for r in random_results)
    print(f"Max random-label train acc: {max_random_acc:.4f}")
    if max_random_acc < 0.90:
        errors.append(f"Expected random-label max train_acc >= 0.90, got {max_random_acc:.4f}")

    # Check that random labels don't generalize
    random_test_accs = [r["test_acc"] for r in random_results]
    mean_random_test = sum(random_test_accs) / len(random_test_accs)
    print(f"Mean random-label test acc: {mean_random_test:.4f}")
    if mean_random_test > 0.30:
        errors.append(f"Random labels should not generalize: mean test_acc={mean_random_test:.4f}")

# Check structured labels
structured_results = [r for r in results if r["label_type"] == "structured"]
if structured_results:
    max_struct_acc = max(r["train_acc"] for r in structured_results)
    print(f"Max structured-label train acc: {max_struct_acc:.4f}")
    if max_struct_acc < 0.90:
        errors.append(f"Expected structured-label max train_acc >= 0.90, got {max_struct_acc:.4f}")

# Check analysis
analysis = data.get("analysis", {})
if not analysis:
    errors.append("Missing analysis section in results")
else:
    for lt in ["random", "structured"]:
        lt_analysis = analysis.get("label_types", {}).get(lt, {})
        sig = lt_analysis.get("sigmoid_fit", {})
        if sig.get("fit_success"):
            print(f"{lt} sigmoid: threshold={sig.get('threshold_params', 'N/A'):.0f} params, "
                  f"sharpness={sig.get('sharpness', 'N/A'):.2f}, "
                  f"R²={sig.get('r_squared', 'N/A'):.4f}")

# Check multi-seed data if present
multi_seed = analysis.get("multi_seed")
if multi_seed:
    aggregated = multi_seed.get("aggregated", [])
    seeds = multi_seed.get("seeds", [])
    print(f"Multi-seed: {len(seeds)} seeds, {len(aggregated)} aggregated entries")
    if len(seeds) < 2:
        errors.append(f"Expected >= 2 seeds, got {len(seeds)}")
    # Check that variance is reported
    for entry in aggregated:
        if entry.get("train_acc_std") is None:
            errors.append(f"Missing train_acc_std for {entry.get('label_type')} h={entry.get('hidden_dim')}")
            break

# Check output files exist
for path in ["results/report.md", "results/figures/memorization_curve.png",
             "results/figures/threshold_comparison.png"]:
    if not os.path.exists(path):
        errors.append(f"Missing output file: {path}")
    else:
        size = os.path.getsize(path)
        print(f"  {path}: {size:,} bytes")

# Final verdict
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
