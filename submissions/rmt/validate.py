"""Validate RMT analysis results for completeness and correctness."""

import json
import os
import sys

# Working-directory guard
if not os.path.isfile("SKILL.md"):
    print("ERROR: validate.py must be executed from the submissions/rmt/ directory.")
    sys.exit(1)

results_path = "results/results.json"
if not os.path.isfile(results_path):
    print(f"ERROR: {results_path} not found. Run run.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

errors = []

# Check metadata
metadata = data.get("metadata", {})
print(f"Seed: {metadata.get('seed')}")
print(f"Hidden dims: {metadata.get('hidden_dims')}")
print(f"Torch version: {metadata.get('torch_version')}")

# Check training results
training = data.get("training_results", [])
n_training = len(training)
expected_models = len(metadata.get("hidden_dims", [])) * 2  # 2 tasks
print(f"\nTraining results: {n_training} (expected {expected_models})")

if n_training != expected_models:
    errors.append(f"Expected {expected_models} training results, got {n_training}")

for tr in training:
    loss = tr.get("final_loss", float("inf"))
    if loss > 10.0:
        errors.append(
            f"{tr['model_label']}: final_loss={loss:.4f} is unusually high"
        )
    if "final_accuracy" in tr and tr["final_accuracy"] < 0.0:
        errors.append(
            f"{tr['model_label']}: negative accuracy {tr['final_accuracy']}"
        )
    if "final_mse" in tr and tr["final_mse"] < 0.0:
        errors.append(
            f"{tr['model_label']}: negative MSE {tr['final_mse']}"
        )

# Check trained analysis
trained = data.get("trained_analysis", [])
untrained = data.get("untrained_analysis", [])
expected_layers = expected_models * 3  # 3 layers per model
print(f"Trained analysis entries: {len(trained)} (expected {expected_layers})")
print(f"Untrained analysis entries: {len(untrained)} (expected {expected_layers})")

if len(trained) != expected_layers:
    errors.append(
        f"Expected {expected_layers} trained analysis entries, got {len(trained)}"
    )
if len(untrained) != expected_layers:
    errors.append(
        f"Expected {expected_layers} untrained analysis entries, got {len(untrained)}"
    )

# Check metric ranges
for entry in trained + untrained:
    label = f"{entry.get('model_label')}/{entry.get('layer_name')}"
    ks = entry.get("ks_statistic", -1)
    if not (0.0 <= ks <= 1.0):
        errors.append(f"{label}: KS statistic {ks:.4f} out of [0, 1]")

    outlier = entry.get("outlier_fraction", -1)
    if not (0.0 <= outlier <= 1.0):
        errors.append(f"{label}: outlier fraction {outlier:.4f} out of [0, 1]")

    snr = entry.get("spectral_norm_ratio", -1)
    if snr < 0:
        errors.append(f"{label}: spectral norm ratio {snr:.4f} is negative")

    kl = entry.get("kl_divergence", -1)
    if kl < 0:
        errors.append(f"{label}: KL divergence {kl:.4f} is negative")

# Core hypothesis: trained models should generally deviate more from MP
# than untrained models (higher average KS statistic)
if trained and untrained:
    avg_trained_ks = sum(e["ks_statistic"] for e in trained) / len(trained)
    avg_untrained_ks = sum(e["ks_statistic"] for e in untrained) / len(untrained)
    print(f"\nAvg KS (trained):   {avg_trained_ks:.4f}")
    print(f"Avg KS (untrained): {avg_untrained_ks:.4f}")

    if avg_trained_ks <= avg_untrained_ks:
        errors.append(
            f"Core hypothesis failed: avg trained KS ({avg_trained_ks:.4f}) "
            f"<= avg untrained KS ({avg_untrained_ks:.4f})"
        )

# Check output files exist
expected_files = [
    "results/results.json",
    "results/report.md",
    "results/eigenvalue_spectra.png",
    "results/ks_summary.png",
    "results/eigenvalue_spectra_untrained.png",
]
for fpath in expected_files:
    if not os.path.isfile(fpath):
        errors.append(f"Missing output file: {fpath}")
    else:
        size = os.path.getsize(fpath)
        print(f"  {fpath}: {size:,} bytes")

# Report
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
