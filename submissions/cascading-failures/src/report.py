"""Report generator: produces a Markdown summary of experiment results."""

from __future__ import annotations

import math
from typing import Dict, List, Any

from src.metrics import (
    aggregate_by_condition,
    topology_ranking,
    hub_vs_random_comparison,
    agent_type_comparison,
)


def _fmt(val: float, decimals: int = 3) -> str:
    """Format a float, returning 'N/A' for inf/nan."""
    if not math.isfinite(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def generate_report(raw_results: List[Dict[str, Any]]) -> str:
    """Generate Markdown report from raw simulation results."""
    aggregated = aggregate_by_condition(raw_results)
    topo_rank = topology_ranking(aggregated)
    hub_vs_rand = hub_vs_random_comparison(aggregated)
    agent_comp = agent_type_comparison(aggregated)

    lines = []
    lines.append("# Cascading Failures in Multi-Agent AI Networks")
    lines.append("")
    lines.append(f"**Total simulations:** {len(raw_results)}")
    lines.append(f"**Unique conditions (aggregated):** {len(aggregated)}")
    lines.append("")

    # Topology ranking
    lines.append("## 1. Topology Systemic Risk Ranking")
    lines.append("")
    lines.append("| Rank | Topology | Mean Systemic Risk | Std |")
    lines.append("|------|----------|--------------------|-----|")
    for i, t in enumerate(topo_rank, 1):
        lines.append(
            f"| {i} | {t['topology']} | {_fmt(t['mean_systemic_risk'])} | "
            f"{_fmt(t['std_systemic_risk'])} |"
        )
    lines.append("")

    # Hub vs random
    lines.append("## 2. Hub vs Random Attack Comparison")
    lines.append("")
    lines.append("| Topology | Hub Cascade | Random Cascade | Hub/Random Ratio |")
    lines.append("|----------|-------------|----------------|------------------|")
    for h in hub_vs_rand:
        lines.append(
            f"| {h['topology']} | {_fmt(h['hub_cascade_size'])} | "
            f"{_fmt(h['random_cascade_size'])} | {_fmt(h['hub_to_random_ratio'], 2)} |"
        )
    lines.append("")

    # Agent type comparison
    lines.append("## 3. Agent Type Resilience")
    lines.append("")
    lines.append("| Rank | Agent Type | Mean Cascade Size | Std |")
    lines.append("|------|------------|-------------------|-----|")
    for i, a in enumerate(agent_comp, 1):
        lines.append(
            f"| {i} | {a['agent_type']} | {_fmt(a['mean_cascade_size'])} | "
            f"{_fmt(a['std_cascade_size'])} |"
        )
    lines.append("")

    # Top-level findings
    lines.append("## 4. Key Findings")
    lines.append("")

    if topo_rank:
        most_risky = topo_rank[0]["topology"]
        least_risky = topo_rank[-1]["topology"]
        lines.append(f"- **Most fragile topology:** {most_risky} "
                      f"(systemic risk = {_fmt(topo_rank[0]['mean_systemic_risk'])})")
        lines.append(f"- **Most resilient topology:** {least_risky} "
                      f"(systemic risk = {_fmt(topo_rank[-1]['mean_systemic_risk'])})")

    if agent_comp:
        best_agent = agent_comp[0]["agent_type"]
        worst_agent = agent_comp[-1]["agent_type"]
        lines.append(f"- **Most resilient agent:** {best_agent} "
                      f"(mean cascade = {_fmt(agent_comp[0]['mean_cascade_size'])})")
        lines.append(f"- **Most fragile agent:** {worst_agent} "
                      f"(mean cascade = {_fmt(agent_comp[-1]['mean_cascade_size'])})")

    # Hub attack finding
    sf_hub = [h for h in hub_vs_rand if h["topology"] == "scale_free"]
    if sf_hub:
        ratio = sf_hub[0]["hub_to_random_ratio"]
        lines.append(f"- **Scale-free hub vulnerability:** hub attacks cause "
                      f"{_fmt(ratio, 2)}x more cascade than random failures")

    lines.append("")
    return "\n".join(lines)
