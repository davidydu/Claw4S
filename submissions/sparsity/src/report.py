"""Generate a Markdown summary report from experiment results."""


def generate_report(results: dict) -> str:
    """Generate a human-readable Markdown report.

    Parameters
    ----------
    results : dict
        Full results from run_all_experiments.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines = []
    lines.append("# Activation Sparsity Evolution During Training")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    cfg = results["config"]
    lines.append(f"- Hidden widths: {cfg['hidden_widths']}")
    lines.append(f"- Epochs: {cfg['n_epochs']}")
    lines.append(f"- Modular addition: lr={cfg['mod_add_lr']}, wd={cfg['mod_add_wd']}")
    lines.append(f"- Regression: lr={cfg['reg_lr']}, wd={cfg['reg_wd']}")
    lines.append(f"- Seed: {cfg['seed']}")
    lines.append("")

    # Per-experiment summary table
    lines.append("## Experiment Results")
    lines.append("")
    lines.append("| Task | Width | Dead% | ZeroFrac | ZeroFrac Change | Test Acc | Gen Gap |")
    lines.append("|------|-------|-------|----------|-----------------|----------|---------|")

    for s in results["experiment_summaries"]:
        task_short = s["task"].replace("modular_addition_mod97", "mod_add").replace(
            "nonlinear_regression", "regression"
        )
        lines.append(
            f"| {task_short} | {s['hidden_dim']} "
            f"| {s['final_dead_frac']:.3f} "
            f"| {s['final_zero_frac']:.3f} "
            f"| {s['zero_frac_change']:+.3f} "
            f"| {s['final_test_acc']:.3f} "
            f"| {s['gen_gap']:.3f} |"
        )
    lines.append("")

    # Correlations
    lines.append("## Sparsity-Generalization Correlations (Spearman)")
    lines.append("")
    corrs = results["correlations"]
    for name, vals in sorted(corrs.items()):
        label = name.replace("_", " ").replace("vs", "vs.")
        sig = "***" if vals["p_value"] < 0.01 else "**" if vals["p_value"] < 0.05 else "*" if vals["p_value"] < 0.1 else ""
        lines.append(f"- **{label}**: rho={vals['rho']:.3f}, p={vals['p_value']:.3f} {sig}")
    lines.append("")

    # Grokking analysis
    lines.append("## Grokking-Sparsity Analysis (Modular Addition)")
    lines.append("")
    for gr in results["grokking_analysis"]:
        lines.append(f"### Width {gr['hidden_dim']}")
        if gr["grokking_detected"]:
            lines.append(f"- Grokking detected at epoch {gr['grokking_epoch']}")
            lines.append(f"- Max test accuracy: {gr['max_test_acc']:.3f}")
            if gr.get("sparsity_at_grokking") is not None:
                lines.append(f"- Dead fraction before grokking: {gr['sparsity_before_grokking']:.3f}")
                lines.append(f"- Dead fraction at grokking: {gr['sparsity_at_grokking']:.3f}")
            if gr.get("zero_frac_at_grokking") is not None:
                lines.append(f"- Zero fraction before grokking: {gr['zero_frac_before_grokking']:.3f}")
                lines.append(f"- Zero fraction at grokking: {gr['zero_frac_at_grokking']:.3f}")
            if gr["sparsity_transition_detected"]:
                lines.append("- **Sparsity transition detected near grokking**")
            else:
                lines.append("- No significant sparsity transition near grokking")
        else:
            lines.append(f"- Grokking not detected within {cfg['n_epochs']} epochs")
            lines.append(f"- Max test accuracy reached: {gr.get('max_test_acc', 'N/A')}")
            if gr.get("zero_frac_initial") is not None:
                lines.append(f"- Zero fraction: {gr['zero_frac_initial']:.3f} -> {gr['zero_frac_final']:.3f}")
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Finding 1: Does sparsity increase during training?
    summaries = results["experiment_summaries"]
    n_increased_zero = sum(1 for s in summaries if s["zero_frac_change"] > 0.01)
    n_decreased_zero = sum(1 for s in summaries if s["zero_frac_change"] < -0.01)
    n_total = len(summaries)
    lines.append(f"1. **Self-sparsification**: {n_increased_zero}/{n_total} experiments showed "
                 f"increased zero activation fraction during training, "
                 f"{n_decreased_zero}/{n_total} showed decreased sparsity.")

    # Finding 2: Width effect
    mod_summaries = [s for s in summaries if "modular" in s["task"]]
    reg_summaries = [s for s in summaries if "regression" in s["task"]]
    if mod_summaries:
        lines.append(f"2. **Width effect (mod add)**: Zero fraction ranges from "
                     f"{min(s['final_zero_frac'] for s in mod_summaries):.3f} to "
                     f"{max(s['final_zero_frac'] for s in mod_summaries):.3f} "
                     f"across widths {[s['hidden_dim'] for s in mod_summaries]}.")
    if reg_summaries:
        lines.append(f"3. **Width effect (regression)**: Zero fraction ranges from "
                     f"{min(s['final_zero_frac'] for s in reg_summaries):.3f} to "
                     f"{max(s['final_zero_frac'] for s in reg_summaries):.3f} "
                     f"across widths {[s['hidden_dim'] for s in reg_summaries]}.")

    # Finding 3: Correlations
    best_corr = max(corrs.items(), key=lambda x: abs(x[1]["rho"]))
    lines.append(f"4. **Strongest correlation**: {best_corr[0].replace('_', ' ')} "
                 f"(rho={best_corr[1]['rho']:.3f}, p={best_corr[1]['p_value']:.3f}).")

    # Finding 4: Grokking and sparsity
    n_grok = sum(1 for gr in results["grokking_analysis"] if gr["grokking_detected"])
    n_trans = sum(1 for gr in results["grokking_analysis"]
                  if gr.get("sparsity_transition_detected", False))
    lines.append(f"5. **Grokking-sparsity coincidence**: {n_grok}/4 widths showed grokking, "
                 f"{n_trans} coincided with sparsity transitions.")

    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- Only 2-layer MLPs studied; deeper architectures may behave differently.")
    lines.append("- Small dataset (modular arithmetic mod 97); real-world tasks have more complexity.")
    lines.append("- Single seed (42); variance across seeds not reported.")
    lines.append("- AdamW weight decay itself promotes sparsity; disentangling optimizer effects requires further study.")
    lines.append("- Per-task hyperparameters (different LR/WD) mean cross-task comparisons confound task with optimizer settings.")
    lines.append("")

    return "\n".join(lines)
