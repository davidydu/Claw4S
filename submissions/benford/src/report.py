"""Markdown report generation for Benford's Law analysis results."""

from src.benford_analysis import benford_expected


def _fmt_dist(dist):
    """Format a digit distribution as a compact table row."""
    digits = range(1, 10)
    vals = [f"{dist.get(str(d), dist.get(d, 0)):.3f}" for d in digits]
    return " | ".join(vals)


def _fmt_benford():
    """Format Benford's expected distribution as a compact table row."""
    expected = benford_expected()
    vals = [f"{expected[d]:.3f}" for d in range(1, 10)]
    return " | ".join(vals)


def generate_report(all_results):
    """Generate a markdown report from all analysis results.

    Args:
        all_results: Dict with keys:
            - "models": dict mapping model_name to epoch-keyed analysis results
            - "controls": dict from generate_control_weights
            - "metadata": dict with runtime info

    Returns:
        Markdown string.
    """
    lines = []
    lines.append("# Benford's Law in Trained Neural Networks")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        "This report analyzes whether the leading digits of trained neural network "
        "weight values follow Benford's Law. We train tiny MLPs on modular arithmetic "
        "and sine regression tasks, saving weight snapshots across training, then test "
        "conformity using chi-squared and Mean Absolute Deviation (MAD) statistics."
    )
    lines.append("")

    # Metadata
    meta = all_results.get("metadata", {})
    if meta:
        lines.append("## Experiment Configuration")
        lines.append("")
        lines.append(f"- **Seed:** {meta.get('seed', 42)}")
        lines.append(f"- **Tasks:** {', '.join(meta.get('tasks', []))}")
        lines.append(f"- **Model sizes:** {', '.join(str(s) for s in meta.get('hidden_sizes', []))}")
        lines.append(f"- **Snapshot epochs:** {meta.get('snapshot_epochs', [])}")
        lines.append(f"- **Runtime:** {meta.get('runtime_seconds', 0):.1f}s")
        lines.append("")

    # Benford reference
    lines.append("## Benford's Law Reference")
    lines.append("")
    lines.append("| Digit | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |")
    lines.append("|-------|---|---|---|---|---|---|---|---|---|")
    lines.append(f"| P(d)  | {_fmt_benford()} |")
    lines.append("")
    lines.append("**MAD classification (Nigrini):** < 0.006 = close, 0.006-0.012 = acceptable, "
                 "0.012-0.015 = marginal, > 0.015 = nonconformity")
    lines.append("")

    # Per-model results
    models = all_results.get("models", {})
    lines.append("## Results by Model")
    lines.append("")

    for model_name, epochs_data in sorted(models.items()):
        lines.append(f"### {model_name}")
        lines.append("")

        # Aggregate MAD over training
        lines.append("#### Training Dynamics (Aggregate Weights)")
        lines.append("")
        lines.append("| Epoch | MAD | Classification | Chi-squared | p-value | N weights |")
        lines.append("|-------|-----|----------------|-------------|---------|-----------|")

        for epoch in sorted(epochs_data.keys(), key=int):
            agg = epochs_data[epoch]["aggregate"]
            lines.append(
                f"| {epoch} | {agg['mad']:.4f} | {agg['mad_class']} | "
                f"{agg['chi2']:.2f} | {agg['p_value']:.4f} | {agg['n_weights']} |"
            )
        lines.append("")

        # Per-layer at final epoch
        final_epoch = max(epochs_data.keys(), key=int)
        per_layer = epochs_data[final_epoch]["per_layer"]

        if per_layer:
            lines.append(f"#### Per-Layer Analysis (Epoch {final_epoch})")
            lines.append("")
            lines.append("| Layer | MAD | Classification | Chi-squared | p-value | N weights |")
            lines.append("|-------|-----|----------------|-------------|---------|-----------|")

            for layer_name in sorted(per_layer.keys()):
                lr = per_layer[layer_name]
                lines.append(
                    f"| {layer_name} | {lr['mad']:.4f} | {lr['mad_class']} | "
                    f"{lr['chi2']:.2f} | {lr['p_value']:.4f} | {lr['n_weights']} |"
                )
            lines.append("")

    # Controls
    controls = all_results.get("controls", {})
    if controls:
        lines.append("## Control Distributions")
        lines.append("")
        lines.append("| Distribution | MAD | Classification | Chi-squared | p-value |")
        lines.append("|-------------|-----|----------------|-------------|---------|")

        for ctrl_name, ctrl_data in sorted(controls.items()):
            lines.append(
                f"| {ctrl_name} | {ctrl_data['mad']:.4f} | {ctrl_data['mad_class']} | "
                f"{ctrl_data['chi2']:.2f} | {ctrl_data['p_value']:.4f} |"
            )
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    _add_findings(lines, models, controls)

    report = "\n".join(lines)
    return report


def _add_findings(lines, models, controls):
    """Add key findings section based on analysis results."""
    findings = []

    # Compare init vs trained
    for model_name, epochs_data in models.items():
        epoch_keys = sorted(epochs_data.keys(), key=int)
        if len(epoch_keys) >= 2:
            first = int(epoch_keys[0])
            last = int(epoch_keys[-1])
            mad_init = epochs_data[epoch_keys[0]]["aggregate"]["mad"]
            mad_final = epochs_data[epoch_keys[-1]]["aggregate"]["mad"]

            if mad_final < mad_init:
                change = "decreased"
                direction = "toward"
            else:
                change = "increased"
                direction = "away from"

            findings.append(
                f"- **{model_name}:** MAD {change} from {mad_init:.4f} (epoch {first}) "
                f"to {mad_final:.4f} (epoch {last}), moving {direction} Benford conformity."
            )

    # Control comparisons
    if "uniform" in controls:
        findings.append(
            f"- **Uniform control:** MAD = {controls['uniform']['mad']:.4f} "
            f"({controls['uniform']['mad_class']}), as expected for a non-Benford distribution."
        )

    if "normal" in controls:
        findings.append(
            f"- **Normal control:** MAD = {controls['normal']['mad']:.4f} "
            f"({controls['normal']['mad_class']})."
        )

    if "kaiming_uniform" in controls:
        findings.append(
            f"- **Kaiming uniform control:** MAD = {controls['kaiming_uniform']['mad']:.4f} "
            f"({controls['kaiming_uniform']['mad_class']})."
        )

    if not findings:
        findings.append("- No specific findings to report.")

    lines.extend(findings)
    lines.append("")
