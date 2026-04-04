# src/report.py
"""Generate markdown report from analysis results."""


def generate_report(analysis_data):
    """Generate a markdown report from analysis results.

    Args:
        analysis_data: dict with 'records' and 'statistics' keys

    Returns:
        str: markdown report
    """
    stats = analysis_data["statistics"]
    records = analysis_data["records"]

    lines = []
    lines.append("# Algorithmic Pricing Collusion: Multi-Agent Auditing Report\n")
    lines.append(f"**Total simulations:** {len(records)}")
    lines.append(f"**Conditions:** {len(stats)}")
    lines.append("")

    # Collusion heatmap (text version)
    lines.append("## Collusion Heatmap\n")
    lines.append("Average collusion score (margin auditor) by matchup × memory:\n")
    lines.append("| Matchup | M=1 | M=3 | M=5 |")
    lines.append("|---------|-----|-----|-----|")

    matchups = sorted(set(s["matchup"] for s in stats))
    for matchup in matchups:
        row = f"| {matchup} "
        for m in [1, 3, 5]:
            entries = [s for s in stats
                       if s["matchup"] == matchup and s["memory"] == m
                       and not s["shocks"]]
            if entries:
                avg = sum(s["avg_auditor_scores"].get("margin", 0)
                          for s in entries) / len(entries)
                row += f"| {avg:.2f} "
            else:
                row += "| — "
        row += "|"
        lines.append(row)

    # Auditor agreement
    lines.append("\n## Auditor Agreement\n")
    lines.append("| Matchup | Memory | Preset | Agreement Rate | Majority | Unanimous |")
    lines.append("|---------|--------|--------|---------------|----------|-----------|")
    for s in sorted(stats, key=lambda x: (-x["majority_collusion_rate"], x["matchup"])):
        if not s["shocks"]:
            lines.append(
                f"| {s['matchup']} | {s['memory']} | {s['preset']} "
                f"| {s['auditor_agreement_rate']:.0%} "
                f"| {s['majority_collusion_rate']:.0%} "
                f"| {s['unanimous_collusion_rate']:.0%} |"
            )

    # Statistical significance
    lines.append("\n## Statistical Tests (prices vs Nash)\n")
    lines.append("| Condition | Avg Price | Nash | Cohen's d | p-value | Significant? |")
    lines.append("|-----------|-----------|------|-----------|---------|-------------|")
    for s in stats:
        if not s["shocks"]:
            sig = "Yes" if s["p_value"] < 0.05 and s["cohens_d"] > 0 else "No"
            lines.append(
                f"| {s['matchup']}/M{s['memory']}/{s['preset']} "
                f"| {s['avg_price']:.3f} | {s['nash_price']:.3f} "
                f"| {s['cohens_d']:.2f} | {s['p_value']:.4f} | {sig} |"
            )

    # Memory effect
    lines.append("\n## Memory Effect\n")
    lines.append("Average margin auditor score by memory length (no-shock runs):\n")
    for m in [1, 3, 5]:
        entries = [s for s in stats if s["memory"] == m and not s["shocks"]]
        if entries:
            avg = sum(s["avg_auditor_scores"].get("margin", 0)
                      for s in entries) / len(entries)
            lines.append(f"- M={m}: {avg:.3f}")

    # Shock robustness
    lines.append("\n## Shock Robustness\n")
    shock_stats = [s for s in stats if s["shocks"]]
    if shock_stats:
        lines.append("| Condition | Majority (no shock) | Majority (with shock) |")
        lines.append("|-----------|--------------------|-----------------------|")
        for s in shock_stats:
            no_shock = [ns for ns in stats
                        if ns["matchup"] == s["matchup"]
                        and ns["memory"] == s["memory"]
                        and ns["preset"] == s["preset"]
                        and not ns["shocks"]]
            ns_rate = no_shock[0]["majority_collusion_rate"] if no_shock else 0
            lines.append(
                f"| {s['matchup']}/M{s['memory']}/{s['preset']} "
                f"| {ns_rate:.0%} | {s['majority_collusion_rate']:.0%} |"
            )

    # Key findings
    lines.append("\n## Key Findings\n")
    # Find highest collusion condition
    no_shock = [s for s in stats if not s["shocks"]]
    if no_shock:
        highest = max(no_shock, key=lambda s: s["avg_auditor_scores"].get("margin", 0))
        lowest = min(no_shock, key=lambda s: s["avg_auditor_scores"].get("margin", 0))
        lines.append(
            f"- **Most collusive condition:** {highest['matchup']}/M{highest['memory']}"
            f"/{highest['preset']} "
            f"(margin score: {highest['avg_auditor_scores'].get('margin', 0):.3f})"
        )
        lines.append(
            f"- **Least collusive condition:** {lowest['matchup']}/M{lowest['memory']}"
            f"/{lowest['preset']} "
            f"(margin score: {lowest['avg_auditor_scores'].get('margin', 0):.3f})"
        )

    lines.append("")
    return "\n".join(lines)


def generate_figures(analysis_data, output_dir="results/figures"):
    """Generate matplotlib figures from analysis results.

    Creates:
    - collusion_heatmap.png: margin auditor score by matchup x memory
    - auditor_agreement.png: agreement rates across conditions
    - memory_effect.png: collusion intensity vs memory length
    """
    import os
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    stats = analysis_data["statistics"]
    no_shock = [s for s in stats if not s["shocks"]]

    # 1. Collusion heatmap (matchup x memory)
    matchups = sorted(set(s["matchup"] for s in no_shock))
    memories = [1, 3, 5]
    heatmap = np.zeros((len(matchups), len(memories)))
    for i, m_name in enumerate(matchups):
        for j, mem in enumerate(memories):
            entries = [s for s in no_shock
                       if s["matchup"] == m_name and s["memory"] == mem]
            if entries:
                heatmap[i, j] = np.mean([
                    s["avg_auditor_scores"].get("margin", 0) for s in entries
                ])

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(heatmap, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(memories)))
    ax.set_xticklabels([f"M={m}" for m in memories])
    ax.set_yticks(range(len(matchups)))
    ax.set_yticklabels(matchups)
    ax.set_xlabel("Memory Length")
    ax.set_ylabel("Agent Matchup")
    ax.set_title("Collusion Intensity (Margin Auditor Score)")
    for i in range(len(matchups)):
        for j in range(len(memories)):
            ax.text(j, i, f"{heatmap[i, j]:.2f}", ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "collusion_heatmap.png"), dpi=150)
    plt.close(fig)

    # 2. Memory effect curve
    fig, ax = plt.subplots(figsize=(7, 4))
    for m_name in matchups:
        scores = []
        for mem in memories:
            entries = [s for s in no_shock
                       if s["matchup"] == m_name and s["memory"] == mem]
            if entries:
                scores.append(np.mean([
                    s["avg_auditor_scores"].get("margin", 0) for s in entries
                ]))
            else:
                scores.append(0)
        ax.plot(memories, scores, marker="o", label=m_name)
    ax.set_xlabel("Memory Length (M)")
    ax.set_ylabel("Avg Margin Auditor Score")
    ax.set_title("Memory Effect on Collusion")
    ax.legend(fontsize=8)
    ax.set_xticks(memories)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "memory_effect.png"), dpi=150)
    plt.close(fig)

    # 3. Auditor agreement
    auditor_names = ["margin", "deviation_punishment", "counterfactual", "welfare"]
    agreement = np.zeros((4, 4))
    count = 0
    for s in no_shock:
        scores = [s["avg_auditor_scores"].get(n, 0) for n in auditor_names]
        verdicts = [sc > 0.5 for sc in scores]
        for a in range(4):
            for b in range(4):
                if verdicts[a] == verdicts[b]:
                    agreement[a, b] += 1
        count += 1
    if count > 0:
        agreement /= count

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(agreement, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(["Margin", "Dev-Punish", "Counter", "Welfare"],
                        rotation=45, ha="right")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Margin", "Dev-Punish", "Counter", "Welfare"])
    ax.set_title("Auditor Pairwise Agreement Rate")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{agreement[i, j]:.0%}", ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "auditor_agreement.png"), dpi=150)
    plt.close(fig)
