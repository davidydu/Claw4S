"""Report generation — Markdown + figures."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from src.experiment import ExperimentResult


def generate_report(results: list[ExperimentResult],
                    aggregated: dict[str, Any],
                    findings: list[dict[str, Any]],
                    output_dir: str = "results") -> str:
    """Generate a Markdown report and save figures.

    Returns the report text.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    lines = [
        "# Data Marketplace Experiment Report",
        "",
        f"**Total simulations:** {aggregated['n_total_simulations']}",
        f"**Unique configurations:** {aggregated['n_groups']}",
        "",
        "## Key Findings",
        "",
    ]

    for i, f in enumerate(findings, 1):
        lines.append(f"### {i}. {f['finding']}")
        lines.append(f"{f['description']}")
        lines.append("")

    # Summary table
    lines.append("## Summary Table (Medium markets, averaged over seeds)")
    lines.append("")
    lines.append("| Composition | Regime | Alloc. Eff. | Surplus/Txn | Lemons | Exploitation |")
    lines.append("|---|---|---|---|---|---|")

    for row in aggregated["summary_table"]:
        if row["market_size"] != "medium":
            continue
        lines.append(
            f"| {row['composition']} | {row['info_regime']} | "
            f"{row['market_efficiency_mean']:.3f} | "
            f"{row['surplus_rate_mean']:.4f} | "
            f"{row['lemons_index_mean']:.3f} | "
            f"{row['audit_exploitation_mean']:.3f} |"
        )
    lines.append("")

    # Buyer surplus per purchase table
    lines.append("## Buyer Surplus per Purchase (Medium, Opaque vs Transparent)")
    lines.append("")
    lines.append("| Composition | Regime | Naive | Reputation | Analytical |")
    lines.append("|---|---|---|---|---|")
    for row in aggregated["summary_table"]:
        if row["market_size"] != "medium" or row["info_regime"] not in ("opaque", "transparent"):
            continue
        bw = row["buyer_welfare"]
        naive_val = f"{bw['naive']:.4f}" if "naive" in bw else "-"
        rep_val = f"{bw['reputation']:.4f}" if "reputation" in bw else "-"
        anal_val = f"{bw['analytical']:.4f}" if "analytical" in bw else "-"
        lines.append(f"| {row['composition']} | {row['info_regime']} | {naive_val} | {rep_val} | {anal_val} |")
    lines.append("")

    report_text = "\n".join(lines)

    # Save report
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Save figures
    _generate_figures(aggregated, output_dir)

    return report_text


def _generate_figures(aggregated: dict[str, Any], output_dir: str) -> None:
    """Generate matplotlib figures and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    rows = aggregated["summary_table"]

    # Figure 1: Market efficiency by composition and regime (medium markets)
    fig, ax = plt.subplots(figsize=(10, 6))
    medium_rows = [r for r in rows if r["market_size"] == "medium"]
    comps = sorted(set(r["composition"] for r in medium_rows))
    regimes = ["transparent", "partial", "opaque"]
    x = np.arange(len(comps))
    width = 0.25

    for i, regime in enumerate(regimes):
        vals = []
        for comp in comps:
            matching = [r for r in medium_rows if r["composition"] == comp and r["info_regime"] == regime]
            vals.append(matching[0]["market_efficiency_mean"] if matching else 0)
        ax.bar(x + i * width, vals, width, label=regime)

    ax.set_ylabel("Market Efficiency")
    ax.set_title("Market Efficiency by Composition and Information Regime")
    ax.set_xticks(x + width)
    ax.set_xticklabels(comps, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.1)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "efficiency_by_composition.png"), dpi=150)
    plt.close(fig)

    # Figure 2: Lemons index by composition and regime
    fig, ax = plt.subplots(figsize=(10, 6))
    for i, regime in enumerate(regimes):
        vals = []
        for comp in comps:
            matching = [r for r in medium_rows if r["composition"] == comp and r["info_regime"] == regime]
            vals.append(matching[0]["lemons_index_mean"] if matching else 0)
        ax.bar(x + i * width, vals, width, label=regime)

    ax.set_ylabel("Lemons Index")
    ax.set_title("Lemons Index by Composition and Information Regime")
    ax.set_xticks(x + width)
    ax.set_xticklabels(comps, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "lemons_by_composition.png"), dpi=150)
    plt.close(fig)

    # Figure 3: Buyer surplus by buyer type across compositions (medium, opaque)
    fig, ax = plt.subplots(figsize=(10, 6))
    opaque_medium = [r for r in rows if r["market_size"] == "medium" and r["info_regime"] == "opaque"]
    comps_opaque = sorted(set(r["composition"] for r in opaque_medium))
    buyer_types = ["naive", "reputation", "analytical"]
    x2 = np.arange(len(comps_opaque))
    width2 = 0.25

    for i, bt in enumerate(buyer_types):
        vals = []
        for comp in comps_opaque:
            matching = [r for r in opaque_medium if r["composition"] == comp]
            if matching:
                vals.append(matching[0]["buyer_surplus"].get(bt, 0))
            else:
                vals.append(0)
        ax.bar(x2 + i * width2, vals, width2, label=bt)

    ax.set_ylabel("Buyer Surplus (value - cost)")
    ax.set_title("Buyer Surplus by Type and Market Composition (Medium, Opaque)")
    ax.set_xticks(x2 + width2)
    ax.set_xticklabels(comps_opaque, rotation=45, ha="right")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "buyer_surplus.png"), dpi=150)
    plt.close(fig)

    # Figure 4: Audit scores heatmap (medium, opaque)
    fig, ax = plt.subplots(figsize=(8, 6))
    audit_keys = ["fair_pricing", "exploitation", "market_efficiency", "information_asymmetry"]
    matrix = []
    for comp in comps_opaque:
        matching = [r for r in opaque_medium if r["composition"] == comp]
        if matching:
            matrix.append([matching[0][f"audit_{k}_mean"] for k in audit_keys])
        else:
            matrix.append([0] * len(audit_keys))

    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(audit_keys)))
    ax.set_xticklabels([k.replace("_", "\n") for k in audit_keys], fontsize=9)
    ax.set_yticks(range(len(comps_opaque)))
    ax.set_yticklabels(comps_opaque)
    ax.set_title("Audit Scores (Medium, Opaque)")
    fig.colorbar(im)

    for i in range(len(comps_opaque)):
        for j in range(len(audit_keys)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "audit_heatmap.png"), dpi=150)
    plt.close(fig)
