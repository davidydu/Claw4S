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

import argparse
import json
import os
import sys
from pathlib import Path

# Guard: must run from submission directory
if not os.path.isfile("run.py"):
    print("ERROR: run.py must be executed from the submission directory.")
    print("  cd submissions/benchmark-difficulty/ && .venv/bin/python run.py")
    sys.exit(1)

from src.analysis import run_full_analysis
from src.plots import (
    plot_feature_correlations,
    plot_difficulty_prediction,
    plot_feature_importance,
)
from src.report import generate_report

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for reproducible analysis runs."""
    parser = argparse.ArgumentParser(
        description="Run benchmark difficulty prediction analysis.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for model training and CV splits (default: 42).",
    )
    parser.add_argument(
        "--use-hardcoded",
        action="store_true",
        help="Use the local hardcoded 98-question sample instead of HF download.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory for output artifacts (default: results).",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=200,
        help="Label permutations for Spearman significance test (default: 200).",
    )
    return parser.parse_args()


args = parse_args()

output_dir = Path(args.output_dir)
figures_dir = output_dir / "figures"
figures_dir.mkdir(parents=True, exist_ok=True)

print("[1/5] Running full analysis...")
results = run_full_analysis(
    use_hardcoded=args.use_hardcoded,
    seed=args.seed,
    n_permutations=args.permutations,
)
print(f"      Analyzed {results['num_questions']} questions, "
      f"{len(results['correlations'])} features")

print("[2/5] Generating correlation plot...")
plot_feature_correlations(
    results["correlations"],
    str(figures_dir / "feature_correlations.png"),
)

print("[3/5] Generating prediction scatter plot...")
plot_difficulty_prediction(
    results["predictions"],
    results["difficulties"],
    str(figures_dir / "difficulty_prediction.png"),
)

print("[4/5] Generating feature importance plot...")
plot_feature_importance(
    results["feature_importances"],
    str(figures_dir / "feature_importance.png"),
)

print("[5/5] Saving results...")

# Save JSON results (exclude non-serializable model object)
json_results = {
    "num_questions": results["num_questions"],
    "correlations": results["correlations"],
    "model_metrics": results["model_metrics"],
    "cv_metrics": results["cv_metrics"],
    "baseline_metrics": results["baseline_metrics"],
    "significance": results["significance"],
    "feature_importances": results["feature_importances"],
    "ranked_features": results["ranked_features"],
    "predictions": results["predictions"],
    "difficulties": results["difficulties"],
    "question_ids": results["question_ids"],
    "data_provenance": results["data_provenance"],
    "seed": results["seed"],
}
with open(output_dir / "results.json", "w") as f:
    json.dump(json_results, f, indent=2)

# Generate and save report
report = generate_report(results)
with open(output_dir / "report.md", "w") as f:
    f.write(report)

print()
print(f"Analysis complete. Results saved to {output_dir}/")
print(f"  {output_dir}/results.json")
print(f"  {output_dir}/report.md")
print(f"  {figures_dir}/feature_correlations.png")
print(f"  {figures_dir}/difficulty_prediction.png")
print(f"  {figures_dir}/feature_importance.png")
print()

provenance = results["data_provenance"]
print("Data provenance:")
print(f"  source:      {provenance['source']}")
print(f"  dataset:     {provenance['dataset_name']}/{provenance['config']}")
print(f"  split:       {provenance['split']}")
print(f"  revision:    {provenance['revision']}")
print()

# Print key metrics
cv = results["cv_metrics"]
baseline = results["baseline_metrics"]
sig = results["significance"]
print("Key metrics (cross-validated):")
print(f"  RF R-squared:        {cv['mean_r_squared']:.4f} +/- {cv['std_r_squared']:.4f}")
print(f"  RF MAE:              {cv['mean_mae']:.4f} +/- {cv['std_mae']:.4f}")
print(f"  RF Spearman (fold):  {cv['mean_spearman']:.4f} +/- {cv['std_spearman']:.4f}")
print(f"  RF Spearman (OOF):   {cv['oof_spearman']:.4f}")
print(f"  Baseline Spearman:   {baseline['oof_spearman']:.4f}")
print(f"  Permutation p-value: {sig['permutation_pvalue']:.4f} "
      f"(n={sig['n_permutations']})")
print()
print("Top 3 features by importance:")
for name, imp in results["ranked_features"][:3]:
    print(f"  {name}: {imp:.4f}")
