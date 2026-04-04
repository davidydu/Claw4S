"""Generate a Markdown report from analysis results."""

from __future__ import annotations


def generate_report(analysis: dict) -> str:
    """Generate a human-readable Markdown report.

    Parameters
    ----------
    analysis : dict
        Output of run_analysis().

    Returns
    -------
    str — Markdown report text.
    """
    meta = analysis["metadata"]
    agg = analysis["aggregated"]
    transitions = analysis["phase_transitions"]
    rankings = analysis["agent_rankings"]

    lines = [
        "# Information Sharing Experiment Report",
        "",
        "## Metadata",
        f"- Simulations: {meta['n_simulations']}",
        f"- Rounds per sim: {meta['n_rounds']}",
        f"- Agents per sim: {meta['n_agents']}",
        f"- Seeds: {meta['seeds']}",
        "",
        "## Agent Type Rankings (by average cumulative payoff)",
        "",
        "| Rank | Type | Avg Payoff | Avg Sharing |",
        "|------|------|-----------|-------------|",
    ]

    for t, info in sorted(rankings.items(), key=lambda x: x[1]["rank"]):
        lines.append(
            f"| {info['rank']} | {t} | {info['avg_payoff']:.1f} | "
            f"{info['avg_sharing']:.3f} |"
        )

    lines.extend([
        "",
        "## Sharing Rates by Condition (tail equilibrium)",
        "",
        "| Composition | Competition | Complementarity | Tail Sharing | Group Welfare |",
        "|------------|-------------|-----------------|-------------|---------------|",
    ])

    for key in sorted(agg.keys()):
        parts = key.split("|")
        d = agg[key]
        lines.append(
            f"| {parts[0]} | {parts[1]} | {parts[2]} | "
            f"{d['tail_sharing_rate']['mean']:.3f} +/- {d['tail_sharing_rate']['std']:.3f} | "
            f"{d['tail_group_welfare']['mean']:.2f} +/- {d['tail_group_welfare']['std']:.2f} |"
        )

    lines.extend([
        "",
        "## Phase Transition Analysis",
        "",
    ])

    for label, info in transitions.items():
        lines.append(f"### {label}")
        levels = info["competition_levels"]
        rates = info["tail_sharing_rates"]
        for l, r in zip(levels, rates):
            lines.append(f"  - competition={l}: tail_sharing={r:.3f}")
        detected = "YES" if info["transition_detected"] else "no"
        lines.append(f"  - Transition detected: **{detected}**")
        lines.append("")

    lines.extend([
        "## Key Findings",
        "",
        _generate_findings(agg, transitions, rankings),
    ])

    report = "\n".join(lines)
    return report


def _generate_findings(agg: dict, transitions: dict, rankings: dict) -> str:
    """Extract key findings from analysis."""
    findings = []

    # 1. Best agent type
    best_type = min(rankings.items(), key=lambda x: x[1]["rank"])
    findings.append(
        f"1. **Best-performing agent type:** {best_type[0]} "
        f"(avg payoff: {best_type[1]['avg_payoff']:.1f})"
    )

    # 2. Sharing vs competition
    for comp in ["mixed", "all_strategic"]:
        low_key = f"{comp}|low|medium"
        high_key = f"{comp}|high|medium"
        if low_key in agg and high_key in agg:
            low_sharing = agg[low_key]["tail_sharing_rate"]["mean"]
            high_sharing = agg[high_key]["tail_sharing_rate"]["mean"]
            findings.append(
                f"2. **{comp} composition:** sharing drops from "
                f"{low_sharing:.3f} (low competition) to "
                f"{high_sharing:.3f} (high competition)"
            )

    # 3. Phase transitions
    n_transitions = sum(1 for v in transitions.values() if v["transition_detected"])
    findings.append(
        f"3. **Phase transitions detected:** {n_transitions}/{len(transitions)} conditions"
    )

    # 4. Group welfare comparison
    open_key = "all_open|medium|medium"
    secret_key = "all_secretive|medium|medium"
    if open_key in agg and secret_key in agg:
        open_welfare = agg[open_key]["tail_group_welfare"]["mean"]
        secret_welfare = agg[secret_key]["tail_group_welfare"]["mean"]
        findings.append(
            f"4. **Cooperation premium:** all-open welfare ({open_welfare:.2f}) vs "
            f"all-secretive ({secret_welfare:.2f}) at medium competition/complementarity"
        )

    return "\n".join(findings)
