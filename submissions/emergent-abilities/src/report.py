"""Generate markdown summary report from analysis results."""

from datetime import datetime, timezone


def generate_report(results: dict) -> str:
    """Generate a markdown report summarizing analysis findings.

    Args:
        results: Output from run_full_analysis().

    Returns:
        Markdown-formatted report string.
    """
    lines = []

    # Header
    lines.append("# Emergent Abilities in LLMs: Mirage or Real?")
    lines.append("")
    lines.append(f"*Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}*")
    lines.append(f"*Random seed: {results['seed']}*")
    lines.append("")

    # Methodology
    lines.append("## Methodology")
    lines.append("")
    lines.append("We re-analyze published benchmark data from BIG-Bench and MMLU to test")
    lines.append("the Schaeffer et al. (2023) claim that emergent abilities in LLMs are")
    lines.append("artifacts of discontinuous evaluation metrics. Our analysis:")
    lines.append("")
    lines.append("1. **Metric Comparison**: Apply both discontinuous (exact match) and")
    lines.append("   continuous (partial credit, token edit distance) metrics to the same data")
    lines.append("2. **Nonlinearity Detection**: Fit sigmoid vs. linear models and compute")
    lines.append("   the Metric Sensitivity Index (MSI)")
    lines.append("3. **Synthetic Demonstration**: Show how linear per-token improvement")
    lines.append("   creates apparent phase transitions under exact-match scoring")
    lines.append("4. **MMLU Scaling**: Analyze smooth scaling under multiple-choice accuracy")
    lines.append("")

    # Key Findings
    lines.append("## Key Findings")
    lines.append("")

    # Finding 1: MSI analysis
    ns = results["nonlinearity_scores"]
    artifact_tasks = [t for t, s in ns.items() if s["msi"] > 2.0]
    genuine_tasks = [t for t, s in ns.items() if s["msi"] <= 2.0]

    lines.append(f"### Finding 1: Metric Sensitivity Index")
    lines.append("")
    lines.append(f"Of {len(ns)} BIG-Bench tasks analyzed:")
    lines.append(f"- **{len(artifact_tasks)}** tasks show MSI > 2.0 (likely metric artifact)")
    lines.append(f"- **{len(genuine_tasks)}** tasks show MSI <= 2.0 (potentially genuine nonlinearity)")
    lines.append("")

    # MSI table
    lines.append("| Task | MSI | Sigmoid R2 (EM) | Linear R2 (EM) | Sigmoid R2 (PC) | Linear R2 (PC) | Verdict |")
    lines.append("|------|-----|-----------------|----------------|-----------------|----------------|---------|")
    for task_name in sorted(ns.keys()):
        s = ns[task_name]
        msi_str = f"{s['msi']:.2f}" if s["msi"] < 100 else ">100"
        verdict = "Artifact" if s["msi"] > 2.0 else "Possibly genuine"
        task_display = task_name.replace("_", " ").title()
        lines.append(
            f"| {task_display} | {msi_str} | "
            f"{s['sigmoid_r2_discontinuous']:.3f} | "
            f"{s['linear_r2_discontinuous']:.3f} | "
            f"{s['sigmoid_r2_continuous']:.3f} | "
            f"{s['linear_r2_continuous']:.3f} | "
            f"{verdict} |"
        )
    lines.append("")

    # Finding 2: Synthetic demo
    demo = results["synthetic_demo"]
    lines.append("### Finding 2: Synthetic Demonstration")
    lines.append("")
    lines.append(f"With {demo['n_tokens']} answer tokens and linearly improving per-token accuracy:")
    lines.append(f"- At lowest model size: per-token acc = {demo['per_token_acc'][0]:.3f}, "
                 f"exact match = {demo['exact_match'][0]:.3f}")
    lines.append(f"- At largest model size: per-token acc = {demo['per_token_acc'][-1]:.3f}, "
                 f"exact match = {demo['exact_match'][-1]:.3f}")
    ratio = demo["per_token_acc"][0] / max(demo["exact_match"][0], 1e-10)
    lines.append(f"- The exact-match metric suppresses performance by {ratio:.1f}x at small scale")
    lines.append("")
    lines.append("This confirms Schaeffer et al.'s core claim: the nonlinear mapping p -> p^n")
    lines.append("creates the appearance of a phase transition from smoothly improving performance.")
    lines.append("")

    # Finding 3: MMLU
    mmlu = results["mmlu_analysis"]
    lines.append("### Finding 3: MMLU Scaling")
    lines.append("")
    lines.append(f"MMLU accuracy (multiple-choice, inherently more continuous) across")
    lines.append(f"{mmlu['n_models']} models:")
    lines.append(f"- Overall linear R2 = {mmlu['overall_linear_r2']:.3f}")
    lines.append(f"- Overall sigmoid R2 = {mmlu['overall_sigmoid_r2']:.3f}")
    lines.append("")

    if mmlu["families"]:
        lines.append("| Family | Models | Linear R2 | Sigmoid R2 | Scaling Pattern |")
        lines.append("|--------|--------|-----------|------------|-----------------|")
        for fam, data in sorted(mmlu["families"].items()):
            pattern = "Sigmoid preferred" if data["prefers_sigmoid"] else "Linear adequate"
            lines.append(
                f"| {fam.upper()} | {len(data['models'])} | "
                f"{data['linear_r2']:.3f} | {data['sigmoid_r2']:.3f} | {pattern} |"
            )
        lines.append("")

    # Finding 4: Metric comparison details
    lines.append("### Finding 4: Metric Comparison Details")
    lines.append("")
    mc = results["metric_comparisons"]
    for task_name, comparison in sorted(mc.items()):
        entries = comparison["entries"]
        if len(entries) < 3:
            continue
        task_display = task_name.replace("_", " ").title()
        lines.append(f"**{task_display}** (n_tokens={comparison['n_tokens']}):")
        lines.append("")
        lines.append("| Model | Params (B) | Exact Match | Partial Credit | Edit Distance |")
        lines.append("|-------|-----------|-------------|----------------|---------------|")
        for e in entries:
            lines.append(
                f"| {e['model']} | {e['params_b']:.1f} | "
                f"{e['exact_match']:.4f} | {e['partial_credit']:.4f} | "
                f"{e['token_edit_distance']:.3f} |"
            )
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")
    lines.append("Our analysis provides evidence consistent with Schaeffer et al. (2023):")
    lines.append("")
    lines.append(f"1. **Most apparent emergence is a metric artifact**: {len(artifact_tasks)}/{len(ns)} tasks")
    lines.append("   show high MSI, meaning the apparent nonlinearity is primarily driven by the")
    lines.append("   discontinuous metric rather than genuine capability jumps.")
    lines.append("")
    if genuine_tasks:
        lines.append(f"2. **Some tasks may show genuine nonlinearity**: {len(genuine_tasks)} tasks")
        lines.append("   retain nonlinear scaling even under continuous metrics, suggesting")
        lines.append("   that not all emergence is an artifact (though sparse data limits conclusions).")
        lines.append("")
    lines.append(f"{'3' if genuine_tasks else '2'}. **MMLU confirms smooth scaling**: With a more continuous metric")
    lines.append("   (multiple-choice accuracy), performance scales relatively smoothly with model size.")
    lines.append("")

    # Limitations
    lines.append("## Limitations and Caveats")
    lines.append("")
    lines.append("1. **Sparse data**: We have only 3-14 model sizes per task, limiting")
    lines.append("   the statistical power of curve-fitting comparisons.")
    lines.append("2. **Token independence assumption**: Our per-token accuracy inference")
    lines.append("   assumes token-level independence, which may not hold for complex tasks.")
    lines.append("3. **Aggregated scores**: We use published accuracy scores, not raw model")
    lines.append("   outputs, so we cannot directly verify the per-token accuracy distribution.")
    lines.append("4. **Cross-paper comparisons**: Scores from different papers may use")
    lines.append("   different evaluation protocols, introducing noise.")
    lines.append("5. **Hardcoded data**: All data is hardcoded from published papers; no")
    lines.append("   model inference was performed.")
    lines.append("")

    # References
    lines.append("## References")
    lines.append("")
    lines.append("- Schaeffer, R., Miranda, B., & Koyejo, S. (2023). Are Emergent Abilities")
    lines.append("  of Large Language Models a Mirage? NeurIPS 2023. arXiv:2304.15004")
    lines.append("- Wei, J., et al. (2022). Emergent Abilities of Large Language Models.")
    lines.append("  arXiv:2206.07682")
    lines.append("- Srivastava, A., et al. (2023). Beyond the Imitation Game: Quantifying")
    lines.append("  and Extrapolating the Capabilities of Language Models. arXiv:2206.04615")
    lines.append("- Hendrycks, D., et al. (2021). Measuring Massive Multitask Language")
    lines.append("  Understanding. ICLR 2021. arXiv:2009.03300")
    lines.append("")

    return "\n".join(lines)
