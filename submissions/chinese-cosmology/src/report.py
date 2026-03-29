# src/report.py
"""Generate markdown report and matplotlib figures from analysis results.

Functions:
  generate_report(analysis_data) → str (markdown)
  generate_figures(analysis_data, output_dir) → None (saves 5 PNGs)
"""

from __future__ import annotations

import os


DOMAINS = ["career", "wealth", "relationships", "health", "overall"]


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def generate_report(analysis_data: dict) -> str:
    """Generate a markdown report from analysis results.

    Args:
        analysis_data: dict returned by analyze_results(), with keys
                       'records', 'evaluator_results', 'statistics'

    Returns:
        str: markdown-formatted report
    """
    stats = analysis_data.get("statistics", {})
    records = analysis_data.get("records", [])
    evaluator_results = analysis_data.get("evaluator_results", {})
    n_records = stats.get("n_records", len(records))

    lines = []
    lines.append("# Cross-System Consistency in Chinese Computational Cosmology\n")
    lines.append(f"**Charts analyzed:** {n_records:,}")
    lines.append("")

    # ---- Cross-system correlation ----
    lines.append("## Cross-System Correlation\n")
    lines.append("Pearson |r| between BaZi and Zi Wei Dou Shu domain scores:\n")
    lines.append("| Domain | BaZi–ZiWei | BaZi–WuXing | ZiWei–WuXing |")
    lines.append("|--------|-----------|------------|-------------|")
    correlation = stats.get("correlation", {})
    for domain in DOMAINS:
        if domain in correlation:
            c = correlation[domain]
            lines.append(
                f"| {domain} "
                f"| {c.get('bazi_ziwei', 0.0):.4f} "
                f"| {c.get('bazi_wuxing', 0.0):.4f} "
                f"| {c.get('ziwei_wuxing', 0.0):.4f} |"
            )
    lines.append("")

    # ---- Correlation inference ----
    correlation_inference = stats.get("correlation_inference", {})
    if correlation_inference:
        pair_labels = {
            "bazi_ziwei": "BaZi–ZiWei",
            "bazi_wuxing": "BaZi–WuXing",
            "ziwei_wuxing": "ZiWei–WuXing",
        }
        lines.append("## Correlation Inference\n")
        lines.append("Fisher-z 95% CI and two-sided p-values for Pearson correlations:")
        lines.append("")
        lines.append("| Domain | Pair | r | 95% CI | p-value | Bonferroni p | n |")
        lines.append("|--------|------|---|--------|---------|--------------|---|")
        for domain in DOMAINS:
            domain_inf = correlation_inference.get(domain, {})
            for pair in ["bazi_ziwei", "bazi_wuxing", "ziwei_wuxing"]:
                inf = domain_inf.get(pair)
                if not inf:
                    continue
                lines.append(
                    f"| {domain} "
                    f"| {pair_labels.get(pair, pair)} "
                    f"| {inf.get('r', 0.0):.4f} "
                    f"| [{inf.get('ci_lower', 0.0):.4f}, {inf.get('ci_upper', 0.0):.4f}] "
                    f"| {inf.get('p_value', 1.0):.3g} "
                    f"| {inf.get('p_value_bonferroni', 1.0):.3g} "
                    f"| {int(inf.get('n', 0)):,} |"
                )
        lines.append("")

    # ---- Domain agreement ----
    lines.append("## Domain Agreement\n")
    lines.append("Fraction of charts where both systems agree on favorability"
                 " (both > 0.5 or both ≤ 0.5):\n")
    lines.append("| Domain | BaZi–ZiWei | BaZi–WuXing | ZiWei–WuXing |")
    lines.append("|--------|-----------|------------|-------------|")
    domain_agreement = stats.get("domain_agreement", {})
    for domain in DOMAINS:
        if domain in domain_agreement:
            a = domain_agreement[domain]
            lines.append(
                f"| {domain} "
                f"| {a.get('bazi_ziwei', 0.0):.4f} "
                f"| {a.get('bazi_wuxing', 0.0):.4f} "
                f"| {a.get('ziwei_wuxing', 0.0):.4f} |"
            )
    lines.append("")

    # ---- Mutual information ----
    lines.append("## Mutual Information (BaZi vs Zi Wei)\n")
    lines.append("Mutual information (nats) between BaZi and Zi Wei domain scores"
                 " with 10-bin discretization:\n")
    mi = stats.get("mutual_information", {})
    for domain in DOMAINS:
        if domain in mi:
            lines.append(f"- **{domain}:** {mi[domain]:.4f} nats")
    lines.append("")

    # ---- Temporal patterns ----
    lines.append("## Temporal Patterns\n")
    lines.append("Mean BaZi–Zi Wei career agreement by year (first 5 years shown):\n")
    temporal = stats.get("temporal_patterns", [])
    lines.append("| Year | Career Agreement |")
    lines.append("|------|-----------------|")
    for yr_data in temporal[:5]:
        lines.append(
            f"| {yr_data['year']} | {yr_data['career_agreement']:.4f} |"
        )
    if len(temporal) > 5:
        lines.append(f"| ... ({len(temporal)} years total) | — |")
    lines.append("")

    # ---- Wu Xing predictiveness ----
    lines.append("## Wu Xing Predictiveness\n")
    lines.append("R² for Wu Xing scores predicting Zi Wei scores per domain:\n")
    for domain in DOMAINS:
        if domain in evaluator_results:
            wx_result = next(
                (r for r in evaluator_results[domain]
                 if r["evaluator_name"] == "wuxing_predictiveness"),
                None,
            )
            if wx_result:
                lines.append(
                    f"- **{domain}:** R² = {wx_result['consistency_score']:.4f}"
                )
    lines.append("")

    # ---- Conditional agreement ----
    cond = stats.get("conditional_agreement", {})
    if cond:
        lines.append("## Conditional Agreement\n")
        lines.append("Agreement rates for extreme (top/bottom 20%) vs. middle charts:\n")
        lines.append(
            f"- **Extreme charts:** {cond.get('extreme_agreement', 0):.4f}"
            f" (n={cond.get('n_extreme', 0):,})"
        )
        lines.append(
            f"- **Middle charts:** {cond.get('middle_agreement', 0):.4f}"
            f" (n={cond.get('n_middle', 0):,})"
        )
        lines.append("")

    # ---- Overall evaluator scores ----
    lines.append("## Evaluator Panel Summary (Career Domain)\n")
    career_evals = evaluator_results.get("career", [])
    if career_evals:
        lines.append("| Evaluator | Consistency Score |")
        lines.append("|-----------|------------------|")
        for r in career_evals:
            lines.append(
                f"| {r['evaluator_name']} | {r['consistency_score']:.4f} |"
            )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(analysis_data: dict, output_dir: str = "results/figures") -> None:
    """Generate 5 matplotlib PNG figures from analysis results.

    Figures:
      1. cross_system_correlation.png  — bar chart of Pearson |r| per domain
      2. domain_agreement.png          — bar chart of agreement rates per domain
      3. mutual_information.png        — MI values with null model comparison placeholder
      4. temporal_patterns.png         — career agreement by year
      5. wuxing_predictiveness.png     — R² for Wu Xing predicting each system

    Args:
        analysis_data: dict returned by analyze_results()
        output_dir:    directory where PNGs are written

    Side effects:
        Creates output_dir if needed; writes 5 PNG files.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)

    stats = analysis_data.get("statistics", {})
    evaluator_results = analysis_data.get("evaluator_results", {})

    # ---- 1. Cross-system correlation ----
    correlation = stats.get("correlation", {})
    pairs = ["bazi_ziwei", "bazi_wuxing", "ziwei_wuxing"]
    pair_labels = ["BaZi–ZiWei", "BaZi–WuXing", "ZiWei–WuXing"]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(DOMAINS))
    width = 0.25
    for j, (pair, label) in enumerate(zip(pairs, pair_labels)):
        vals = [correlation.get(d, {}).get(pair, 0.0) for d in DOMAINS]
        ax.bar(x + j * width, vals, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("Pearson |r|")
    ax.set_title("Cross-System Correlation by Domain")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "cross_system_correlation.png"), dpi=150)
    plt.close(fig)

    # ---- 2. Domain agreement ----
    domain_agreement = stats.get("domain_agreement", {})

    fig, ax = plt.subplots(figsize=(9, 5))
    for j, (pair, label) in enumerate(zip(pairs, pair_labels)):
        vals = [domain_agreement.get(d, {}).get(pair, 0.0) for d in DOMAINS]
        ax.bar(x + j * width, vals, width, label=label)
    ax.set_xticks(x + width)
    ax.set_xticklabels(DOMAINS)
    ax.set_ylabel("Agreement Rate")
    ax.set_title("Domain Agreement Rate (Both >0.5 or Both ≤0.5)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", label="chance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "domain_agreement.png"), dpi=150)
    plt.close(fig)

    # ---- 3. Mutual information ----
    mi = stats.get("mutual_information", {})
    mi_vals = [mi.get(d, 0.0) for d in DOMAINS]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(DOMAINS, mi_vals, color="steelblue")
    ax.set_ylabel("Mutual Information (nats)")
    ax.set_title("BaZi–Zi Wei Mutual Information by Domain")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "mutual_information.png"), dpi=150)
    plt.close(fig)

    # ---- 4. Temporal patterns ----
    temporal = stats.get("temporal_patterns", [])
    if temporal:
        years = [t["year"] for t in temporal]
        agreements = [t["career_agreement"] for t in temporal]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(years, agreements, marker=".", markersize=4, linewidth=1)
        ax.axhline(0.5, color="grey", linestyle="--", label="chance")
        ax.set_xlabel("Year")
        ax.set_ylabel("Career Agreement Rate")
        ax.set_title("BaZi–Zi Wei Career Agreement by Year")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "temporal_patterns.png"), dpi=150)
        plt.close(fig)
    else:
        # Write a blank figure if no temporal data
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No temporal data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("Temporal Patterns (no data)")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "temporal_patterns.png"), dpi=150)
        plt.close(fig)

    # ---- 5. Wu Xing predictiveness ----
    wx_r2 = {}
    for domain in DOMAINS:
        evals = evaluator_results.get(domain, [])
        wx = next(
            (r for r in evals if r["evaluator_name"] == "wuxing_predictiveness"),
            None,
        )
        wx_r2[domain] = wx["consistency_score"] if wx else 0.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(DOMAINS, [wx_r2[d] for d in DOMAINS], color="darkorange")
    ax.set_ylabel("R²")
    ax.set_title("Wu Xing Predictiveness (R²) by Domain")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "wuxing_predictiveness.png"), dpi=150)
    plt.close(fig)
