"""Generate markdown report from RMT analysis results."""

import os


def generate_report(results_data: dict) -> str:
    """Generate a human-readable markdown report.

    Args:
        results_data: Full results dict with metadata, training_results,
                      trained_analysis, and untrained_analysis.

    Returns:
        Markdown-formatted report string.
    """
    metadata = results_data["metadata"]
    training = results_data["training_results"]
    trained = results_data["trained_analysis"]
    untrained = results_data["untrained_analysis"]

    lines = []
    lines.append("# Random Matrix Theory Analysis of Neural Network Weights")
    lines.append("")
    lines.append(f"Seed: {metadata['seed']}")
    lines.append("")

    # Training summary
    lines.append("## Training Summary")
    lines.append("")
    lines.append("| Model | Task | Hidden Dim | Final Loss | Metric |")
    lines.append("|-------|------|-----------|------------|--------|")
    for tr in training:
        metric_str = ""
        if "final_accuracy" in tr:
            metric_str = f"Accuracy: {tr['final_accuracy']:.4f}"
        elif "final_mse" in tr:
            metric_str = f"MSE: {tr['final_mse']:.6f}"
        lines.append(
            f"| {tr['model_label']} | {tr['task']} | {tr['hidden_dim']} | "
            f"{tr['final_loss']:.4f} | {metric_str} |"
        )
    lines.append("")

    # RMT Analysis: Trained models
    lines.append("## RMT Analysis: Trained Models")
    lines.append("")
    lines.append(
        "| Model | Layer | Shape | KS Stat | Outlier Frac | "
        "Spectral Ratio | KL Div |"
    )
    lines.append("|-------|-------|-------|---------|-------------|----------------|--------|")
    for r in trained:
        lines.append(
            f"| {r['model_label']} | {r['layer_name']} | "
            f"{r['shape'][0]}x{r['shape'][1]} | "
            f"{r['ks_statistic']:.4f} | {r['outlier_fraction']:.4f} | "
            f"{r['spectral_norm_ratio']:.4f} | {r['kl_divergence']:.4f} |"
        )
    lines.append("")

    # RMT Analysis: Untrained (random init) baselines
    lines.append("## RMT Analysis: Untrained Baselines (Random Init)")
    lines.append("")
    lines.append(
        "| Model | Layer | Shape | KS Stat | Outlier Frac | "
        "Spectral Ratio | KL Div |"
    )
    lines.append("|-------|-------|-------|---------|-------------|----------------|--------|")
    for r in untrained:
        lines.append(
            f"| {r['model_label']} | {r['layer_name']} | "
            f"{r['shape'][0]}x{r['shape'][1]} | "
            f"{r['ks_statistic']:.4f} | {r['outlier_fraction']:.4f} | "
            f"{r['spectral_norm_ratio']:.4f} | {r['kl_divergence']:.4f} |"
        )
    lines.append("")

    # Comparison: trained vs untrained
    lines.append("## Trained vs Untrained Comparison")
    lines.append("")
    lines.append("| Model | Layer | KS (Trained) | KS (Untrained) | Delta KS |")
    lines.append("|-------|-------|-------------|----------------|----------|")

    trained_by_key = {
        (r["model_label"], r["layer_name"]): r for r in trained
    }
    untrained_by_key = {
        (r["model_label"], r["layer_name"]): r for r in untrained
    }

    delta_ks_values = []
    for key in sorted(trained_by_key.keys()):
        t = trained_by_key[key]
        u = untrained_by_key.get(key)
        if u:
            delta = t["ks_statistic"] - u["ks_statistic"]
            delta_ks_values.append(delta)
            lines.append(
                f"| {key[0]} | {key[1]} | "
                f"{t['ks_statistic']:.4f} | {u['ks_statistic']:.4f} | "
                f"{delta:+.4f} |"
            )
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    findings = []

    # Find layer with max KS in trained models
    if trained:
        max_ks_trained = max(trained, key=lambda r: r["ks_statistic"])
        findings.append(
            f"**Largest MP deviation (trained):** {max_ks_trained['model_label']} / "
            f"{max_ks_trained['layer_name']} with KS={max_ks_trained['ks_statistic']:.4f}"
        )

    if untrained:
        max_ks_untrained = max(untrained, key=lambda r: r["ks_statistic"])
        findings.append(
            f"**Largest MP deviation (untrained):** {max_ks_untrained['model_label']} / "
            f"{max_ks_untrained['layer_name']} with KS={max_ks_untrained['ks_statistic']:.4f}"
        )

    if delta_ks_values:
        avg_delta = sum(delta_ks_values) / len(delta_ks_values)
        std_delta = (
            sum((d - avg_delta) ** 2 for d in delta_ks_values)
            / max(1, len(delta_ks_values) - 1)
        ) ** 0.5
        findings.append(
            f"**Average KS increase after training:** "
            f"{avg_delta:+.4f} (std={std_delta:.4f})"
        )
        positive_deltas = sum(1 for d in delta_ks_values if d > 0)
        findings.append(
            f"**Layers with increased deviation:** {positive_deltas}/{len(delta_ks_values)}"
        )
        nonpositive_keys = [
            key for key in sorted(trained_by_key.keys())
            if trained_by_key[key]["ks_statistic"] <=
            untrained_by_key[key]["ks_statistic"]
        ]
        if nonpositive_keys:
            all_degenerate = all(
                trained_by_key[key].get("n_eigenvalues", 0) <= 1
                for key in nonpositive_keys
            )
            if all_degenerate:
                findings.append(
                    f"**Layers without increased deviation:** "
                    f"{len(nonpositive_keys)}/{len(delta_ks_values)}, "
                    "all single-eigenvalue layers where MP comparisons "
                    "are degenerate."
                )
            else:
                nonpositive_labels = ", ".join(
                    f"{model} / {layer}"
                    for model, layer in nonpositive_keys
                )
                findings.append(
                    f"**Layers without increased deviation:** "
                    f"{len(nonpositive_keys)}/{len(delta_ks_values)} "
                    f"({nonpositive_labels})"
                )

    # Spectral norm analysis
    trained_snr = [r["spectral_norm_ratio"] for r in trained]
    untrained_snr = [r["spectral_norm_ratio"] for r in untrained]
    if trained_snr and untrained_snr:
        avg_trained_snr = sum(trained_snr) / len(trained_snr)
        avg_untrained_snr = sum(untrained_snr) / len(untrained_snr)
        findings.append(
            f"**Avg spectral norm ratio:** trained={avg_trained_snr:.3f}, "
            f"untrained={avg_untrained_snr:.3f}"
        )

    # Task-level comparison
    mod_trained = [r for r in trained if "mod" in r.get("model_label", "")]
    reg_trained = [r for r in trained if "regression" in r.get("model_label", "")]
    if mod_trained and reg_trained:
        avg_mod_ks = sum(r["ks_statistic"] for r in mod_trained) / len(mod_trained)
        avg_reg_ks = sum(r["ks_statistic"] for r in reg_trained) / len(reg_trained)
        findings.append(
            f"**Avg KS by task:** modular arithmetic={avg_mod_ks:.4f}, "
            f"regression={avg_reg_ks:.4f}"
        )

    for index, finding in enumerate(findings, start=1):
        lines.append(f"{index}. {finding}")

    lines.append("")
    lines.append("## Methodology Notes")
    lines.append("")
    lines.append(
        "- Eigenvalues computed from correlation matrix C = (1/M) W^T W "
        "where W is the weight matrix of shape (M, N)."
    )
    lines.append(
        "- MP parameters: gamma = min(M,N)/max(M,N), "
        "sigma^2 = empirical variance of W entries."
    )
    lines.append(
        "- KS statistic measures maximum distance between empirical and "
        "theoretical CDFs."
    )
    lines.append(
        "- Outlier fraction counts eigenvalues outside the MP bulk "
        "[lambda_-, lambda_+]."
    )
    lines.append(
        "- All models trained with seed=42 for reproducibility."
    )
    lines.append(
        "- **Finite-size effects:** Layers with very few eigenvalues "
        "(e.g., regression fc1 with N=3) show elevated KS even for untrained "
        "weights because the MP law is an asymptotic result (M, N -> infinity). "
        "Comparisons are most meaningful for layers with N >= 32."
    )
    lines.append(
        "- **Degenerate layers:** Layers with N=1 (regression fc3) are excluded "
        "from MP analysis as a single eigenvalue cannot form a distribution."
    )
    lines.append("")

    return "\n".join(lines)


def save_report(report: str, path: str = "results/report.md") -> str:
    """Save report to file.

    Args:
        report: Markdown report string.
        path: Output file path.

    Returns:
        Path to saved file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
    return path
