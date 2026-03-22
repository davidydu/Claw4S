"""Validate double descent analysis results for completeness and correctness."""

import json
import os
import sys

# Working-directory guard
if not os.path.exists(os.path.join("src", "sweep.py")):
    print(
        "ERROR: validate.py must be run from the submissions/double-descent/ directory.\n"
        "  cd submissions/double-descent/ && .venv/bin/python validate.py"
    )
    sys.exit(1)

errors = []

# Check results.json exists and loads
results_path = "results/results.json"
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run run.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

# Check metadata
meta = data.get("metadata", {})
print(f"Training samples:     {meta.get('n_train', '?')}")
print(f"Test samples:         {meta.get('n_test', '?')}")
print(f"Input dimensions:     {meta.get('d', '?')}")
print(f"Noise levels:         {meta.get('noise_levels', '?')}")
print(f"RF widths tested:     {len(meta.get('rf_widths', []))}")
print(f"MLP widths tested:    {len(meta.get('mlp_widths', []))}")
print(f"Variance seeds:       {meta.get('variance_seeds', '?')}")
print(f"Runtime:              {meta.get('runtime_seconds', '?'):.1f}s")
print()

# Check runtime
runtime = meta.get("runtime_seconds", 999)
if runtime > 180:
    errors.append(f"Runtime {runtime:.0f}s exceeds 3-minute limit")
else:
    print(f"Runtime OK ({runtime:.1f}s < 180s)")

# Check random features results
rf = data.get("random_features", {})
if len(rf) < 2:
    errors.append(f"Expected >= 2 noise levels in random_features, got {len(rf)}")
else:
    print(f"Random features noise levels: {len(rf)}")

for label, results in rf.items():
    if len(results) < 10:
        errors.append(f"random_features[{label}] has only {len(results)} widths")
    test_losses = [r["test_loss"] for r in results]
    peak_loss = max(test_losses)
    min_loss = min(test_losses)
    ratio = peak_loss / max(min_loss, 1e-10)
    noise_val = label.replace("noise_", "sigma=")
    print(f"  {noise_val}: {len(results)} widths, peak/min ratio={ratio:.1f}x")

# Check for double descent detection
highest_noise = f"noise_{max(meta.get('noise_levels', [1.0]))}"
if highest_noise in rf:
    results = rf[highest_noise]
    test_losses = [r["test_loss"] for r in results]
    peak = max(test_losses)
    final = test_losses[-1]
    if final < peak * 0.1:
        print(f"  Double descent confirmed: final loss << peak loss")
    else:
        errors.append("Double descent not clearly visible in highest noise condition")

# Check MLP results
mlp = data.get("mlp_sweep", [])
if len(mlp) < 5:
    errors.append(f"MLP sweep has only {len(mlp)} widths, expected >= 5")
else:
    print(f"MLP widths tested: {len(mlp)}")

# Check epoch-wise results
epoch_wise = data.get("epoch_wise", {})
if len(epoch_wise) < 1:
    errors.append("No epoch-wise results found")
else:
    print(f"Epoch-wise noise levels: {len(epoch_wise)}")
    for label, result in epoch_wise.items():
        n_epochs = len(result.get("epochs", []))
        noise_val = label.replace("noise_", "sigma=")
        print(f"  {noise_val}: {n_epochs} epoch checkpoints")

# Check variance results
variance = data.get("variance", [])
if len(variance) < 2:
    errors.append(f"Variance has only {len(variance)} seeds, expected >= 2")
else:
    print(f"Variance seeds: {len(variance)}")

# Check plot files
expected_plots = [
    "results/model_wise_double_descent.png",
    "results/noise_comparison.png",
    "results/epoch_wise_double_descent.png",
    "results/mlp_comparison.png",
    "results/variance_bands.png",
]
for path in expected_plots:
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        print(f"  {path}: {size_kb:.0f} KB")
    else:
        errors.append(f"Missing plot: {path}")

# Check report
if os.path.exists("results/report.md"):
    with open("results/report.md") as f:
        report_lines = len(f.readlines())
    print(f"  results/report.md: {report_lines} lines")
else:
    errors.append("Missing results/report.md")

print()

if errors:
    print(f"Validation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation passed.")
