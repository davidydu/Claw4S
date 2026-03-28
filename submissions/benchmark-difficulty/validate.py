"""Validate analysis results for completeness and correctness."""

import json
import os
import sys

if not os.path.exists("src/data.py"):
    print("ERROR: Must run from submissions/benchmark-difficulty/ directory")
    raise SystemExit(1)

from src.data import (
    EASY2HARD_DATASET_NAME,
    EASY2HARD_CONFIG,
    EASY2HARD_SPLIT,
    EASY2HARD_DATASET_REVISION,
)


errors = []

# Check results.json exists and is valid
if not os.path.isfile("results/results.json"):
    print("ERROR: results/results.json not found. Run run.py first.")
    sys.exit(1)

with open("results/results.json") as f:
    data = json.load(f)

num_questions = data["num_questions"]
num_features = len(data["correlations"])
num_predictions = len(data["predictions"])
num_difficulties = len(data["difficulties"])

if num_questions < 1000:
    print(f"WARNING: Only {num_questions} questions found — fallback hardcoded "
          f"data was likely used instead of the full Easy2Hard-Bench dataset.")

print(f"Questions analyzed: {num_questions}")
print(f"Features computed:  {num_features}")
print(f"Predictions:        {num_predictions}")
print(f"Difficulties:       {num_difficulties}")

# Validate counts
if num_questions < 50:
    errors.append(f"Expected >= 50 questions, got {num_questions}")
if num_features < 10:
    errors.append(f"Expected >= 10 features, got {num_features}")
if num_predictions != num_questions:
    errors.append(f"Predictions count ({num_predictions}) != questions ({num_questions})")
if num_difficulties != num_questions:
    errors.append(f"Difficulties count ({num_difficulties}) != questions ({num_questions})")

# Validate correlations
print("\nFeature correlations (Spearman rho):")
for name, corr in sorted(data["correlations"].items(),
                          key=lambda x: abs(x[1]["rho"]), reverse=True):
    rho = corr["rho"]
    pval = corr["pvalue"]
    sig = "*" if pval < 0.05 else " "
    print(f"  {sig} {name:30s}  rho={rho:+.4f}  p={pval:.4f}")
    if not (-1.0 <= rho <= 1.0):
        errors.append(f"Invalid rho for {name}: {rho}")
    if not (0.0 <= pval <= 1.0):
        errors.append(f"Invalid p-value for {name}: {pval}")

# Validate model metrics
mm = data["model_metrics"]
print(f"\nModel metrics (train):")
print(f"  R-squared: {mm['r_squared']:.4f}")
print(f"  MAE:       {mm['mae']:.4f}")

if mm["r_squared"] < 0.0:
    errors.append(f"Negative R-squared on training data: {mm['r_squared']:.4f}")
if mm["mae"] > 0.5:
    errors.append(f"MAE too high on training data: {mm['mae']:.4f}")

# Validate cross-validation metrics
cv = data["cv_metrics"]
print(f"\nCross-validation metrics:")
print(f"  R-squared:   {cv['mean_r_squared']:.4f} +/- {cv['std_r_squared']:.4f}")
print(f"  MAE:         {cv['mean_mae']:.4f} +/- {cv['std_mae']:.4f}")
print(f"  Spearman:    {cv['mean_spearman']:.4f} +/- {cv['std_spearman']:.4f}")
if "oof_spearman" in cv:
    print(f"  Spearman OOF:{cv['oof_spearman']:.4f}")
    if not (-1.0 <= cv["oof_spearman"] <= 1.0):
        errors.append(f"Invalid CV OOF Spearman: {cv['oof_spearman']}")
else:
    errors.append("Missing cv_metrics.oof_spearman")

if len(cv["fold_scores"]) < 3:
    errors.append(f"Expected >= 3 CV folds, got {len(cv['fold_scores'])}")

# Validate baseline metrics
baseline = data.get("baseline_metrics")
if baseline is None:
    errors.append("Missing baseline_metrics in results.json")
else:
    required_baseline_keys = {
        "mean_r_squared",
        "std_r_squared",
        "mean_mae",
        "std_mae",
        "mean_spearman",
        "std_spearman",
        "oof_spearman",
        "fold_scores",
    }
    missing = required_baseline_keys - set(baseline.keys())
    if missing:
        errors.append(f"Missing baseline metric keys: {sorted(missing)}")
    else:
        print("\nBaseline metrics (dummy mean predictor):")
        print(f"  R-squared:   {baseline['mean_r_squared']:.4f} +/- "
              f"{baseline['std_r_squared']:.4f}")
        print(f"  MAE:         {baseline['mean_mae']:.4f} +/- "
              f"{baseline['std_mae']:.4f}")
        print(f"  Spearman:    {baseline['mean_spearman']:.4f} +/- "
              f"{baseline['std_spearman']:.4f}")
        print(f"  Spearman OOF:{baseline['oof_spearman']:.4f}")
        if len(baseline["fold_scores"]) != len(cv["fold_scores"]):
            errors.append("Baseline fold count does not match CV fold count")

# Validate significance testing
sig = data.get("significance")
if sig is None:
    errors.append("Missing significance section in results.json")
else:
    required_sig_keys = {"oof_spearman", "permutation_pvalue", "n_permutations"}
    missing = required_sig_keys - set(sig.keys())
    if missing:
        errors.append(f"Missing significance keys: {sorted(missing)}")
    else:
        print("\nPermutation significance:")
        print(f"  OOF Spearman:       {sig['oof_spearman']:.4f}")
        print(f"  Permutation pvalue: {sig['permutation_pvalue']:.4f}")
        print(f"  # permutations:     {sig['n_permutations']}")
        if not (-1.0 <= sig["oof_spearman"] <= 1.0):
            errors.append(f"Invalid OOF Spearman: {sig['oof_spearman']}")
        if not (0.0 <= sig["permutation_pvalue"] <= 1.0):
            errors.append(f"Invalid permutation p-value: {sig['permutation_pvalue']}")
        if sig["n_permutations"] < 0:
            errors.append(f"Invalid n_permutations: {sig['n_permutations']}")

# Validate provenance metadata
prov = data.get("data_provenance")
if prov is None:
    errors.append("Missing data_provenance in results.json")
else:
    required_prov_keys = {
        "source",
        "dataset_name",
        "config",
        "split",
        "revision",
        "is_fallback",
    }
    missing = required_prov_keys - set(prov.keys())
    if missing:
        errors.append(f"Missing provenance keys: {sorted(missing)}")
    else:
        print("\nData provenance:")
        print(f"  source:   {prov['source']}")
        print(f"  dataset:  {prov['dataset_name']}/{prov['config']}")
        print(f"  split:    {prov['split']}")
        print(f"  revision: {prov['revision']}")
        if prov["dataset_name"] != EASY2HARD_DATASET_NAME:
            errors.append("Unexpected dataset_name in provenance")
        if prov["config"] != EASY2HARD_CONFIG:
            errors.append("Unexpected config in provenance")
        if prov["split"] != EASY2HARD_SPLIT:
            errors.append("Unexpected split in provenance")
        if prov["revision"] != EASY2HARD_DATASET_REVISION:
            errors.append("Dataset revision is not pinned to expected commit")
        if prov["source"].startswith("hardcoded"):
            print("WARNING: Hardcoded dataset sample was used.")

# Validate feature importances
fi = data["feature_importances"]
total_imp = sum(fi.values())
print(f"\nFeature importances sum: {total_imp:.4f}")
if abs(total_imp - 1.0) > 0.05:
    errors.append(f"Feature importances don't sum to 1.0: {total_imp:.4f}")

# Validate figures exist
figure_files = [
    "results/figures/feature_correlations.png",
    "results/figures/difficulty_prediction.png",
    "results/figures/feature_importance.png",
]
for fig_path in figure_files:
    if not os.path.isfile(fig_path):
        errors.append(f"Missing figure: {fig_path}")
    else:
        size = os.path.getsize(fig_path)
        if size < 1000:
            errors.append(f"Figure too small ({size} bytes): {fig_path}")

# Validate report exists
if not os.path.isfile("results/report.md"):
    errors.append("Missing report: results/report.md")
else:
    with open("results/report.md") as f:
        report_text = f.read()
    if len(report_text) < 500:
        errors.append(f"Report too short: {len(report_text)} chars")

# Final verdict
if errors:
    print(f"\nValidation FAILED with {len(errors)} error(s):")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("\nValidation passed.")
