"""Generate markdown report from experiment results."""

from typing import Any


def generate_report(results: dict[str, Any]) -> str:
    """Generate a human/agent-readable markdown report.

    Args:
        results: Full results dict from run_experiment().

    Returns:
        Markdown-formatted report string.
    """
    meta = results["metadata"]
    aggs = results["aggregates"]
    findings = results["findings"]

    lines = [
        "# Shortcut Learning Detection: Results Report",
        "",
        "## Experiment Configuration",
        f"- **Configurations:** {meta['n_configs']} "
        f"({len(meta['hidden_dims'])} widths x {len(meta['weight_decays'])} weight decays x {len(meta['seeds'])} seeds)",
        f"- **Hidden dims:** {meta['hidden_dims']}",
        f"- **Weight decays:** {meta['weight_decays']}",
        f"- **Genuine features:** {meta['n_genuine_features']}, "
        f"Total features: {meta['n_total_features']} (last = shortcut)",
        f"- **Training samples:** {meta['n_train']}, Test samples: {meta['n_test']}",
        f"- **Runtime:** {meta['elapsed_seconds']}s",
        "",
        "## Results Table",
        "",
        "| Hidden | Weight Decay | Train Acc | Test (w/ shortcut) | Test (w/o shortcut) | Shortcut Reliance |",
        "|--------|-------------|-----------|-------------------|--------------------|--------------------|",
    ]

    for a in aggs:
        lines.append(
            f"| {a['hidden_dim']:>6} | {a['weight_decay']:>11} | "
            f"{a['train_acc_mean']:.3f} +/- {a['train_acc_std']:.3f} | "
            f"{a['test_acc_with_mean']:.3f} +/- {a['test_acc_with_std']:.3f} | "
            f"{a['test_acc_without_mean']:.3f} +/- {a['test_acc_without_std']:.3f} | "
            f"{a['shortcut_reliance_mean']:.3f} +/- {a['shortcut_reliance_std']:.3f} |"
        )

    lines.append("")
    lines.append("## Key Findings")
    lines.append("")
    for i, f in enumerate(findings, 1):
        lines.append(f"{i}. {f}")

    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "Shortcut reliance is defined as `test_acc_with_shortcut - test_acc_without_shortcut`. "
        "A value near zero means the model relies on genuine features; a large positive value "
        "indicates dependence on the spurious shortcut feature."
    )
    lines.append("")

    report = "\n".join(lines)
    return report
