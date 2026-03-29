"""Validate experiment results for completeness and scientific soundness.

Must be executed from the submissions/dp-membership/ directory.
Checks: results exist, all privacy levels present, metrics in valid ranges,
thesis supported (strong DP reduces AUC toward 0.5).
"""

import json
import os
import sys

from src.runtime import ensure_submission_cwd

# Working directory guard
ensure_submission_cwd(__file__)

# Load results
results_path = "results/results.json"
if not os.path.exists(results_path):
    print(f"ERROR: {results_path} not found. Run run.py first.")
    sys.exit(1)

with open(results_path) as f:
    data = json.load(f)

errors = []
warnings = []

# --- Check metadata ---
meta = data.get("metadata", {})
required_meta_keys = [
    "n_privacy_levels",
    "n_seeds",
    "seeds",
    "n_samples",
    "n_features",
    "n_classes",
    "hidden_dim",
    "epochs",
    "batch_size",
    "lr",
    "n_shadows",
    "max_grad_norm",
    "cluster_std",
    "delta",
    "total_runs",
    "elapsed_seconds",
]
for key in required_meta_keys:
    if key not in meta:
        errors.append(f"Missing metadata key: {key}")

n_levels = meta.get("n_privacy_levels", 0)
n_seeds = meta.get("n_seeds", 0)
n_runs = meta.get("total_runs", 0)

print(f"Privacy levels: {n_levels}")
print(f"Seeds: {n_seeds}")
print(f"Total runs: {n_runs} (expected {n_levels * n_seeds})")

if n_levels != 4:
    errors.append(f"Expected 4 privacy levels, got {n_levels}")
if n_seeds < 3:
    errors.append(f"Expected >= 3 seeds, got {n_seeds}")
if n_runs != n_levels * n_seeds:
    errors.append(f"Expected {n_levels * n_seeds} total runs, got {n_runs}")

# --- Check per-trial results ---
per_trial = data.get("per_trial", [])
if len(per_trial) != n_runs:
    errors.append(f"Expected {n_runs} trial results, got {len(per_trial)}")

seen_pairs = set()
level_to_seeds: dict[str, set[int]] = {}
for trial in per_trial:
    level = trial.get("privacy_level")
    seed = trial.get("seed")
    pair = (level, seed)
    if pair in seen_pairs:
        errors.append(f"Duplicate trial for level={level}, seed={seed}")
    seen_pairs.add(pair)
    level_to_seeds.setdefault(level, set()).add(seed)

    auc = trial.get("attack_auc", -1)
    if not (0.0 <= auc <= 1.0):
        errors.append(f"Invalid attack AUC {auc} for {trial['privacy_level']} seed={trial['seed']}")

    test_acc = trial.get("test_accuracy", -1)
    if not (0.0 <= test_acc <= 1.0):
        errors.append(f"Invalid test accuracy {test_acc} for {trial['privacy_level']} seed={trial['seed']}")

# --- Check aggregated results ---
agg = data.get("aggregated", {})
expected_levels = ["non-private", "weak-dp", "moderate-dp", "strong-dp"]
for level in expected_levels:
    if level not in agg:
        errors.append(f"Missing aggregated results for {level}")
    seeds_for_level = level_to_seeds.get(level, set())
    if len(seeds_for_level) != n_seeds:
        errors.append(
            f"Expected {n_seeds} unique seeds for {level}, got {len(seeds_for_level)}"
        )

# --- Check epsilon ordering consistency ---
if all(level in agg for level in ["weak-dp", "moderate-dp", "strong-dp"]):
    weak_eps = agg["weak-dp"]["metrics"]["epsilon"]["mean"]
    moderate_eps = agg["moderate-dp"]["metrics"]["epsilon"]["mean"]
    strong_eps = agg["strong-dp"]["metrics"]["epsilon"]["mean"]
    print(f"DP epsilon means: weak={weak_eps:.2f}, moderate={moderate_eps:.2f}, strong={strong_eps:.2f}")
    if not (weak_eps > moderate_eps > strong_eps):
        errors.append(
            "Expected epsilon ordering weak-dp > moderate-dp > strong-dp, "
            f"got weak={weak_eps:.2f}, moderate={moderate_eps:.2f}, strong={strong_eps:.2f}"
        )

# --- Check thesis: strong DP should reduce AUC toward 0.5 ---
if "non-private" in agg and "strong-dp" in agg:
    np_auc = agg["non-private"]["metrics"]["attack_auc"]["mean"]
    sd_auc = agg["strong-dp"]["metrics"]["attack_auc"]["mean"]
    print(f"\nNon-private attack AUC:  {np_auc:.3f}")
    print(f"Strong-DP attack AUC:    {sd_auc:.3f}")
    print(f"AUC reduction:           {np_auc - sd_auc:.3f}")

    if sd_auc >= np_auc:
        errors.append(f"Strong DP AUC ({sd_auc:.3f}) >= non-private AUC ({np_auc:.3f}); expected reduction")

    if sd_auc > 0.65:
        warnings.append(f"Strong DP AUC ({sd_auc:.3f}) is above 0.65; expected closer to 0.5")

# --- Check non-private model has reasonable accuracy ---
if "non-private" in agg:
    np_acc = agg["non-private"]["metrics"]["test_accuracy"]["mean"]
    print(f"Non-private test accuracy: {np_acc:.3f}")
    if np_acc < 0.4:
        warnings.append(f"Non-private test accuracy ({np_acc:.3f}) seems low; check data/model")

# --- Check plots exist ---
expected_plots = [
    "results/attack_auc_vs_privacy.png",
    "results/privacy_utility_leakage.png",
    "results/generalization_gap_vs_attack.png",
]
for plot_path in expected_plots:
    if os.path.exists(plot_path):
        print(f"Plot exists: {plot_path}")
    else:
        errors.append(f"Missing plot: {plot_path}")

# --- Report ---
print()
if warnings:
    print(f"Warnings ({len(warnings)}):")
    for w in warnings:
        print(f"  - {w}")

if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Validation PASSED.")
