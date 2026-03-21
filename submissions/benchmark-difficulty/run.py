"""Run the full benchmark difficulty prediction analysis.

This script orchestrates the complete pipeline:
  1. Load ARC-Challenge questions with IRT difficulty scores
  2. Extract structural features from each question
  3. Compute feature-difficulty correlations
  4. Train and cross-validate a Random Forest difficulty model
  5. Generate plots and a summary report
  6. Save all results to results/

Expected runtime: < 1 minute on CPU.
"""

import json
import os
import sys

# Guard: must run from submission directory
if not os.path.isfile("run.py"):
    print("ERROR: run.py must be executed from the submission directory.")
    print("  cd submissions/benchmark-difficulty/ && .venv/bin/python run.py")
    sys.exit(1)

os.makedirs("results/figures", exist_ok=True)

from src.analysis import run_full_analysis
from src.plots import (
    plot_feature_correlations,
    plot_difficulty_prediction,
    plot_feature_importance,
)
from src.report import generate_report

SEED = 42

print("[1/5] Running full analysis...")
results = run_full_analysis(use_hardcoded=False, seed=SEED)
print(f"      Analyzed {results['num_questions']} questions, "
      f"{len(results['correlations'])} features")

print("[2/5] Generating correlation plot...")
plot_feature_correlations(
    results["correlations"],
    "results/figures/feature_correlations.png",
)

print("[3/5] Generating prediction scatter plot...")
plot_difficulty_prediction(
    results["predictions"],
    results["difficulties"],
    "results/figures/difficulty_prediction.png",
)

print("[4/5] Generating feature importance plot...")
plot_feature_importance(
    results["feature_importances"],
    "results/figures/feature_importance.png",
)

print("[5/5] Saving results...")

# Save JSON results (exclude non-serializable model object)
json_results = {
    "num_questions": results["num_questions"],
    "correlations": results["correlations"],
    "model_metrics": results["model_metrics"],
    "cv_metrics": results["cv_metrics"],
    "feature_importances": results["feature_importances"],
    "ranked_features": results["ranked_features"],
    "predictions": results["predictions"],
    "difficulties": results["difficulties"],
    "question_ids": results["question_ids"],
    "seed": results["seed"],
}
with open("results/results.json", "w") as f:
    json.dump(json_results, f, indent=2)

# Generate and save report
report = generate_report(results)
with open("results/report.md", "w") as f:
    f.write(report)

print()
print("Analysis complete. Results saved to results/")
print(f"  results/results.json")
print(f"  results/report.md")
print(f"  results/figures/feature_correlations.png")
print(f"  results/figures/difficulty_prediction.png")
print(f"  results/figures/feature_importance.png")
print()

# Print key metrics
cv = results["cv_metrics"]
print(f"Key metrics (cross-validated):")
print(f"  R-squared:   {cv['mean_r_squared']:.4f} +/- {cv['std_r_squared']:.4f}")
print(f"  MAE:         {cv['mean_mae']:.4f} +/- {cv['std_mae']:.4f}")
print(f"  Spearman:    {cv['mean_spearman']:.4f} +/- {cv['std_spearman']:.4f}")
print()
print(f"Top 3 features by importance:")
for name, imp in results["ranked_features"][:3]:
    print(f"  {name}: {imp:.4f}")
