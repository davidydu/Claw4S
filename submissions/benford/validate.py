"""Validate Benford's Law analysis results for completeness and correctness."""

import json
import os
import sys

RESULTS_FILE = "results/results.json"
FIGURES_DIR = "results/figures"
REPORT_FILE = "results/report.md"

# Working-directory guard: ensure we are running from submissions/benford/
if not os.path.exists("src/benford_analysis.py"):
    print(
        "ERROR: src/benford_analysis.py not found. "
        "Run validate.py from the submissions/benford/ directory."
    )
    sys.exit(1)

errors = []

# Check results file exists
if not os.path.exists(RESULTS_FILE):
    print(f"ERROR: {RESULTS_FILE} not found. Run run.py first.")
    sys.exit(1)

with open(RESULTS_FILE) as f:
    data = json.load(f)

# Check metadata
meta = data.get("metadata", {})
print(f"Seed: {meta.get('seed', 'MISSING')}")
print(f"Tasks: {meta.get('tasks', [])}")
print(f"Hidden sizes: {meta.get('hidden_sizes', [])}")
print(f"Snapshot epochs: {meta.get('snapshot_epochs', [])}")
print(f"Epochs: {meta.get('epochs', 'MISSING')}")
print(f"Learning rate: {meta.get('learning_rate', 'MISSING')}")
print(f"Quick mode: {meta.get('quick_mode', False)}")
print(f"Plots enabled: {meta.get('make_plots', True)}")
print(f"Runtime: {meta.get('runtime_seconds', 0):.1f}s")

required_meta = [
    "python_version",
    "torch_version",
    "numpy_version",
    "scipy_version",
    "matplotlib_version",
    "epochs",
    "learning_rate",
    "controls_n",
    "quick_mode",
    "make_plots",
]
for field in required_meta:
    if field not in meta:
        errors.append(f"Missing metadata field: {field}")

if meta.get("runtime_seconds", 0) <= 0:
    errors.append("runtime_seconds must be > 0")

# Check models
models = data.get("models", {})
num_models = len(models)
print(f"\nModels analyzed: {num_models}")

expected_model_count = 2 * len(meta.get("hidden_sizes", []))
if expected_model_count > 0 and num_models != expected_model_count:
    errors.append(f"Expected {expected_model_count} models, got {num_models}")
elif expected_model_count == 0 and num_models < 1:
    errors.append("Expected at least 1 model")

expected_epochs = set(str(e) for e in meta.get("snapshot_epochs", []))

for model_name, epochs_data in models.items():
    actual_epochs = set(epochs_data.keys())
    missing = expected_epochs - actual_epochs
    if missing:
        errors.append(f"{model_name}: missing snapshot epochs {missing}")

    for epoch, result in epochs_data.items():
        agg = result.get("aggregate", {})
        if not agg:
            errors.append(f"{model_name} epoch {epoch}: no aggregate results")
            continue

        # Check required fields
        for field in ["n_weights", "chi2", "p_value", "mad", "mad_class", "observed_dist"]:
            if field not in agg:
                errors.append(f"{model_name} epoch {epoch}: missing {field}")

        # Sanity checks
        if "n_weights" in agg and agg["n_weights"] < 10:
            errors.append(
                f"{model_name} epoch {epoch}: too few weights ({agg['n_weights']})"
            )

        if "mad" in agg:
            mad = agg["mad"]
            if not (0 <= mad <= 1.0):
                errors.append(
                    f"{model_name} epoch {epoch}: MAD {mad:.4f} out of range [0, 1]"
                )

        if "p_value" in agg:
            p = agg["p_value"]
            if not (0 <= p <= 1.0):
                errors.append(
                    f"{model_name} epoch {epoch}: p-value {p:.4f} out of range [0, 1]"
                )

        if "observed_dist" in agg:
            dist = agg["observed_dist"]
            if len(dist) != 9:
                errors.append(
                    f"{model_name} epoch {epoch}: expected 9 digits, got {len(dist)}"
                )
            total = sum(dist.values())
            if abs(total - 1.0) > 0.01:
                errors.append(
                    f"{model_name} epoch {epoch}: digit proportions sum to {total:.4f}"
                )

    # Print summary for this model
    first_epoch = min(epochs_data.keys(), key=int)
    last_epoch = max(epochs_data.keys(), key=int)
    mad_init = epochs_data[first_epoch]["aggregate"].get("mad", -1)
    mad_final = epochs_data[last_epoch]["aggregate"].get("mad", -1)
    print(
        f"  {model_name}: MAD {mad_init:.4f} (epoch {first_epoch}) -> "
        f"{mad_final:.4f} (epoch {last_epoch})"
    )

# Check controls
controls = data.get("controls", {})
num_controls = len(controls)
print(f"\nControls: {num_controls}")

if num_controls < 2:
    errors.append(f"Expected >= 2 controls, got {num_controls}")

for ctrl_name, ctrl_data in controls.items():
    for field in ["n_weights", "chi2", "p_value", "mad", "mad_class"]:
        if field not in ctrl_data:
            errors.append(f"Control {ctrl_name}: missing {field}")
    print(f"  {ctrl_name}: MAD = {ctrl_data.get('mad', -1):.4f} ({ctrl_data.get('mad_class', '?')})")

# Check figures
expected_figures = [
    "controls_comparison.png",
]
for model_name in models:
    expected_figures.extend([
        f"{model_name}_digits.png",
        f"{model_name}_mad_training.png",
        f"{model_name}_layers.png",
    ])

missing_figures = []
if meta.get("make_plots", True):
    for fig in expected_figures:
        fig_path = os.path.join(FIGURES_DIR, fig)
        if not os.path.exists(fig_path):
            missing_figures.append(fig)

    if missing_figures:
        errors.append(f"Missing figures: {missing_figures}")
    else:
        print(f"\nFigures: all {len(expected_figures)} present")
else:
    print("\nFigures: skipped (make_plots=False)")

# Check report
if not os.path.exists(REPORT_FILE):
    errors.append(f"Report file {REPORT_FILE} not found")
else:
    with open(REPORT_FILE) as f:
        report_content = f.read()
    if len(report_content) < 200:
        errors.append(f"Report too short ({len(report_content)} chars)")
    else:
        print(f"\nReport: {len(report_content)} chars")

# Final verdict
print()
if errors:
    print(f"Validation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation passed.")
