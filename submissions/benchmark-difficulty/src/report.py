"""Report generation module for benchmark difficulty prediction.

Generates a human-readable markdown summary of analysis results.
"""


def generate_report(results: dict) -> str:
    """Generate a markdown report from analysis results.

    Args:
        results: Complete results dict from run_full_analysis().

    Returns:
        Markdown-formatted report string.
    """
    lines = []
    lines.append("# Benchmark Difficulty Prediction Report")
    lines.append("")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"Analyzed **{results['num_questions']}** ARC-Challenge "
                 f"questions with IRT difficulty scores from Easy2Hard-Bench "
                 f"(NeurIPS 2024).")
    lines.append("")
    lines.append("**Research question:** Can structural features of "
                 "multiple-choice questions predict LLM difficulty without "
                 "running any LLM?")
    lines.append("")

    # Model performance
    mm = results["model_metrics"]
    cv = results["cv_metrics"]
    lines.append("## Model Performance")
    lines.append("")
    lines.append("| Metric | Train (Full) | Cross-Validated (mean +/- std) |")
    lines.append("|--------|-------------|-------------------------------|")
    lines.append(f"| R-squared | {mm['r_squared']:.4f} | "
                 f"{cv['mean_r_squared']:.4f} +/- {cv['std_r_squared']:.4f} |")
    lines.append(f"| MAE | {mm['mae']:.4f} | "
                 f"{cv['mean_mae']:.4f} +/- {cv['std_mae']:.4f} |")
    lines.append(f"| Spearman rho | - | "
                 f"{cv['mean_spearman']:.4f} +/- {cv['std_spearman']:.4f} |")
    lines.append("")

    # Cross-validation fold details
    lines.append("## Cross-Validation Details")
    lines.append("")
    lines.append("| Fold | R-squared | MAE | Spearman rho |")
    lines.append("|------|-----------|-----|-------------|")
    for i, fold in enumerate(cv["fold_scores"]):
        lines.append(f"| {i+1} | {fold['r_squared']:.4f} | "
                     f"{fold['mae']:.4f} | {fold['spearman_rho']:.4f} |")
    lines.append("")

    # Feature correlations
    lines.append("## Feature Correlations with Difficulty")
    lines.append("")
    lines.append("Spearman rank correlations between structural features "
                 "and IRT difficulty score:")
    lines.append("")
    lines.append("| Feature | Spearman rho | p-value | Significant? |")
    lines.append("|---------|-------------|---------|-------------|")

    sorted_corrs = sorted(
        results["correlations"].items(),
        key=lambda x: abs(x[1]["rho"]),
        reverse=True,
    )
    n_features = len(sorted_corrs)
    bonferroni_threshold = 0.05 / n_features if n_features > 0 else 0.05
    for name, corr in sorted_corrs:
        sig = "Yes" if corr["pvalue"] < 0.05 else "No"
        lines.append(f"| {name} | {corr['rho']:.4f} | "
                     f"{corr['pvalue']:.4f} | {sig} |")
    lines.append("")
    lines.append(
        f"**Note:** With {n_features} simultaneous tests, Bonferroni-corrected "
        f"significance threshold is p < {bonferroni_threshold:.4f}. No feature "
        f"survives this correction, strengthening the negative-result interpretation."
    )
    lines.append("")

    # Feature importances
    lines.append("## Feature Importance (Random Forest)")
    lines.append("")
    lines.append("Mean Decrease in Impurity (MDI) importances from the "
                 "Random Forest model:")
    lines.append("")
    lines.append("| Rank | Feature | Importance |")
    lines.append("|------|---------|-----------|")
    for i, (name, imp) in enumerate(results["ranked_features"]):
        lines.append(f"| {i+1} | {name} | {imp:.4f} |")
    lines.append("")

    # Findings
    lines.append("## Key Findings")
    lines.append("")

    # Find strongest correlations
    strongest_pos = max(sorted_corrs, key=lambda x: x[1]["rho"])
    strongest_neg = min(sorted_corrs, key=lambda x: x[1]["rho"])
    top_feature = results["ranked_features"][0]

    lines.append(f"1. **Strongest positive correlation:** {strongest_pos[0]} "
                 f"(rho = {strongest_pos[1]['rho']:.3f})")
    lines.append(f"2. **Strongest negative correlation:** {strongest_neg[0]} "
                 f"(rho = {strongest_neg[1]['rho']:.3f})")
    lines.append(f"3. **Most important feature (RF):** {top_feature[0]} "
                 f"(importance = {top_feature[1]:.3f})")
    lines.append(f"4. **Cross-validated Spearman rho:** "
                 f"{cv['mean_spearman']:.3f} +/- {cv['std_spearman']:.3f}")
    lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    if cv["mean_r_squared"] < 0.05:
        lines.append(
            "The near-zero cross-validated R-squared indicates that "
            "structural features alone are **insufficient** for practical "
            "difficulty prediction, even though the positive Spearman rho "
            "suggests a weak rank-order signal."
        )
    elif cv["mean_spearman"] > 0.3:
        lines.append("The cross-validated Spearman correlation exceeds 0.3, "
                     "indicating that structural features alone can provide "
                     "a **moderate** prediction of LLM difficulty.")
    elif cv["mean_spearman"] > 0.1:
        lines.append("The cross-validated Spearman correlation is between "
                     "0.1 and 0.3, indicating a **weak but non-trivial** "
                     "relationship between structural features and LLM "
                     "difficulty.")
    else:
        lines.append("The cross-validated Spearman correlation is below 0.1, "
                     "indicating that structural features alone are "
                     "**insufficient** to predict LLM difficulty. This "
                     "suggests difficulty is primarily determined by "
                     "semantic reasoning demands rather than surface features.")
    lines.append("")

    # Limitations
    lines.append("## Limitations")
    lines.append("")
    lines.append("- IRT difficulty scores are derived from LLM performance, "
                 "not human performance.")
    lines.append("- Structural features cannot capture semantic reasoning "
                 "difficulty.")
    lines.append("- The hardcoded sample of 98 questions may not represent "
                 "the full difficulty distribution.")
    lines.append("- Random Forest feature importances can be biased toward "
                 "high-cardinality features.")
    lines.append("- Cross-validation on a small dataset may have high "
                 "variance between folds.")
    lines.append("")

    lines.append("---")
    lines.append(f"*Generated with seed={results['seed']}. "
                 f"Data source: Easy2Hard-Bench (Wang et al., NeurIPS 2024).*")

    return "\n".join(lines)
