"""Report generator for information cascade experiment results."""

from __future__ import annotations


def _fmt(val: float | None, digits: int = 3) -> str:
    if val is None:
        return "N/A"
    return f"{val:.{digits}f}"


def _fmt_ci(ci: tuple[float | None, float | None] | None, digits: int = 3) -> str:
    if ci is None or ci[0] is None:
        return ""
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


def generate_report(metrics: list[dict], metadata: dict) -> str:
    """Generate a markdown report from computed metrics.

    Args:
        metrics: List of metric dicts from compute_all_metrics().
        metadata: Dict with runtime info (n_sims, runtime_s, etc.).

    Returns:
        Markdown string.
    """
    lines: list[str] = []
    lines.append("# Information Cascade Experiment Report")
    lines.append("")
    lines.append(f"- Total simulations: {metadata['n_simulations']}")
    lines.append(f"- Runtime: {metadata['runtime_s']:.1f}s")
    lines.append(f"- Agent types: {metadata['n_agent_types']}")
    lines.append(f"- Signal qualities: {metadata['signal_qualities']}")
    lines.append(f"- Sequence lengths: {metadata['sequence_lengths']}")
    lines.append("")

    # Table 1: Cascade Formation Rate by agent type x signal quality (averaged over N)
    lines.append("## Cascade Formation Rate")
    lines.append("")
    lines.append("Fraction of simulations where an information cascade formed,")
    lines.append("averaged across sequence lengths and seeds.")
    lines.append("")

    # Group by agent_type x signal_quality
    by_type_sq: dict[tuple[str, float], list[dict]] = {}
    for m in metrics:
        key = (m["agent_type"], m["signal_quality"])
        by_type_sq.setdefault(key, []).append(m)

    sqs = sorted(set(m["signal_quality"] for m in metrics))
    agent_types = sorted(set(m["agent_type"] for m in metrics))
    labels = {m["agent_type"]: m["agent_label"] for m in metrics}

    header = "| Agent | " + " | ".join(f"q={q}" for q in sqs) + " |"
    sep = "|---|" + "|".join("---" for _ in sqs) + "|"
    lines.append(header)
    lines.append(sep)
    for at in agent_types:
        row = f"| {labels[at]} |"
        for sq in sqs:
            group = by_type_sq.get((at, sq), [])
            if group:
                avg_rate = sum(m["cascade_formation_rate"] for m in group) / len(group)
                row += f" {avg_rate:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")

    # Table 2: Cascade Accuracy
    lines.append("## Cascade Accuracy")
    lines.append("")
    lines.append("Fraction of formed cascades that matched the true state.")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for at in agent_types:
        row = f"| {labels[at]} |"
        for sq in sqs:
            group = by_type_sq.get((at, sq), [])
            accs = [m["cascade_accuracy"] for m in group if m["cascade_accuracy"] is not None]
            if accs:
                avg_acc = sum(accs) / len(accs)
                row += f" {avg_acc:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")

    # Table 3: Cascade Fragility
    lines.append("## Cascade Fragility")
    lines.append("")
    lines.append("Fraction of formed cascades that were broken before sequence end.")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for at in agent_types:
        row = f"| {labels[at]} |"
        for sq in sqs:
            group = by_type_sq.get((at, sq), [])
            frags = [m["cascade_fragility"] for m in group if m["cascade_fragility"] is not None]
            if frags:
                avg_frag = sum(frags) / len(frags)
                row += f" {avg_frag:.3f} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")

    # Table 4: Mean Cascade Length
    lines.append("## Mean Cascade Length")
    lines.append("")
    lines.append("Average number of consecutive agents following the cascade action.")
    lines.append("")
    lines.append(header)
    lines.append(sep)
    for at in agent_types:
        row = f"| {labels[at]} |"
        for sq in sqs:
            group = by_type_sq.get((at, sq), [])
            lens = [m["mean_cascade_length"] for m in group if m["mean_cascade_length"] is not None]
            if lens:
                avg_len = sum(lens) / len(lens)
                row += f" {avg_len:.1f} |"
            else:
                row += " N/A |"
        lines.append(row)
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Find highest/lowest formation rates
    all_rates = [(m["agent_label"], m["signal_quality"], m["cascade_formation_rate"]) for m in metrics]
    if all_rates:
        # Average formation rate per agent type
        by_type_rate: dict[str, list[float]] = {}
        for label, sq, rate in all_rates:
            by_type_rate.setdefault(label, []).append(rate)
        avg_by_type = {k: sum(v) / len(v) for k, v in by_type_rate.items()}
        highest = max(avg_by_type, key=avg_by_type.get)  # type: ignore[arg-type]
        lowest = min(avg_by_type, key=avg_by_type.get)  # type: ignore[arg-type]
        lines.append(f"- Highest cascade formation rate: **{highest}** ({avg_by_type[highest]:.3f})")
        lines.append(f"- Lowest cascade formation rate: **{lowest}** ({avg_by_type[lowest]:.3f})")

        # Effect of signal quality on formation (averaged across types)
        by_sq_rate: dict[float, list[float]] = {}
        for _, sq, rate in all_rates:
            by_sq_rate.setdefault(sq, []).append(rate)
        for sq in sorted(by_sq_rate.keys()):
            avg = sum(by_sq_rate[sq]) / len(by_sq_rate[sq])
            lines.append(f"- Signal quality q={sq}: avg formation rate = {avg:.3f}")

    lines.append("")
    return "\n".join(lines)
