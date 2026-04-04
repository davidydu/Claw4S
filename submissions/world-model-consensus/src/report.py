"""
Report generation: Markdown summary and matplotlib figures.
"""

from __future__ import annotations

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.analysis import AggregatedMetric, detect_phase_transition, compute_sharpness


def generate_figures(
    results_by_key: Dict[Tuple[str, int, float], Dict[str, AggregatedMetric]],
    output_dir: Path,
) -> List[str]:
    """Generate all figures and return list of file paths."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figures: List[str] = []

    # ---- Figure 1: Coordination rate vs disagreement (one line per composition, N=4) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    compositions = sorted(set(k[0] for k in results_by_key))
    n_target = 4

    for comp in compositions:
        ds, means, stds = [], [], []
        for (c, n, d), metrics in sorted(results_by_key.items()):
            if c == comp and n == n_target and "coordination_rate" in metrics:
                ds.append(d)
                means.append(metrics["coordination_rate"].mean)
                stds.append(metrics["coordination_rate"].std)
        if ds:
            ax.errorbar(ds, means, yerr=stds, marker="o", label=comp, capsize=3)

    ax.set_xlabel("Disagreement Level", fontsize=12)
    ax.set_ylabel("Coordination Rate (final 20%)", fontsize=12)
    ax.set_title("Coordination Rate vs Prior Disagreement (N=4)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    path = output_dir / "fig1_coordination_vs_disagreement.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    # ---- Figure 2: Consensus time vs disagreement ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for comp in compositions:
        ds, means = [], []
        for (c, n, d), metrics in sorted(results_by_key.items()):
            if c == comp and n == n_target and "consensus_time" in metrics:
                ds.append(d)
                means.append(metrics["consensus_time"].mean)
        if ds:
            ax.plot(ds, means, marker="s", label=comp)

    ax.set_xlabel("Disagreement Level", fontsize=12)
    ax.set_ylabel("Rounds to Sustained Consensus", fontsize=12)
    ax.set_title("Consensus Speed vs Prior Disagreement (N=4)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    path = output_dir / "fig2_consensus_time.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    # ---- Figure 3: Coordination rate vs disagreement for different group sizes (adaptive only) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    group_sizes = sorted(set(k[1] for k in results_by_key))

    for n in group_sizes:
        ds, means, stds = [], [], []
        for (c, n2, d), metrics in sorted(results_by_key.items()):
            if c == "all_adaptive" and n2 == n and "coordination_rate" in metrics:
                ds.append(d)
                means.append(metrics["coordination_rate"].mean)
                stds.append(metrics["coordination_rate"].std)
        if ds:
            ax.errorbar(ds, means, yerr=stds, marker="o", label=f"N={n}", capsize=3)

    ax.set_xlabel("Disagreement Level", fontsize=12)
    ax.set_ylabel("Coordination Rate", fontsize=12)
    ax.set_title("Group Size Effect on Coordination (all-Adaptive)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    path = output_dir / "fig3_group_size_effect.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    # ---- Figure 4: Fairness vs disagreement ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for comp in compositions:
        ds, means = [], []
        for (c, n, d), metrics in sorted(results_by_key.items()):
            if c == comp and n == n_target and "fairness" in metrics:
                val = metrics["fairness"].mean
                if val >= 0:  # skip -1 (undefined)
                    ds.append(d)
                    means.append(val)
        if ds:
            ax.plot(ds, means, marker="^", label=comp)

    ax.set_xlabel("Disagreement Level", fontsize=12)
    ax.set_ylabel("Majority-Preference Fraction", fontsize=12)
    ax.set_title("Fairness: Whose Preference Wins? (N=4)", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    path = output_dir / "fig4_fairness.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figures.append(str(path))

    return figures


def generate_markdown_report(
    results_by_key: Dict[Tuple[str, int, float], Dict[str, AggregatedMetric]],
    output_dir: Path,
    figures: List[str],
) -> str:
    """Generate a Markdown report and save to output_dir/report.md."""

    lines: List[str] = []
    lines.append("# World Model Consensus — Experiment Report\n")
    lines.append("## 1. Overview\n")
    lines.append(f"- **Simulations run:** {len(results_by_key)} unique conditions")
    lines.append("- **Rounds per sim:** 10,000")
    lines.append("- **Seeds:** 3 per condition")
    lines.append("- **Metrics:** coordination_rate, consensus_time, welfare, fairness\n")

    # Phase transition table
    lines.append("## 2. Phase Transition Detection\n")
    lines.append("| Composition | N | Transition Point | Sharpness |")
    lines.append("|-------------|---|-----------------|-----------|")

    compositions = sorted(set(k[0] for k in results_by_key))
    group_sizes = sorted(set(k[1] for k in results_by_key))

    for comp in compositions:
        for n in group_sizes:
            ds, rates = [], []
            for (c, n2, d), metrics in sorted(results_by_key.items()):
                if c == comp and n2 == n and "coordination_rate" in metrics:
                    ds.append(d)
                    rates.append(metrics["coordination_rate"].mean)
            if ds:
                tp = detect_phase_transition(ds, rates)
                sharp = compute_sharpness(ds, rates)
                tp_str = f"{tp:.3f}" if tp is not None else "none"
                lines.append(f"| {comp} | {n} | {tp_str} | {sharp:.3f} |")

    # Coordination rate table (N=4)
    lines.append("\n## 3. Coordination Rate Table (N=4)\n")
    header = "| Composition | " + " | ".join(f"d={d}" for d in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]) + " |"
    lines.append(header)
    lines.append("|" + "---|" * 8)
    for comp in compositions:
        row = f"| {comp} "
        for d in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
            key = (comp, 4, d)
            if key in results_by_key and "coordination_rate" in results_by_key[key]:
                m = results_by_key[key]["coordination_rate"]
                row += f"| {m.mean:.3f}+/-{m.std:.3f} "
            else:
                row += "| — "
        row += "|"
        lines.append(row)

    # Figures
    lines.append("\n## 4. Figures\n")
    for f in figures:
        fname = Path(f).name
        lines.append(f"![{fname}]({fname})\n")

    # Key findings (placeholder — filled by analysis)
    lines.append("\n## 5. Key Findings\n")
    lines.append("*(auto-generated from analysis)*\n")

    # Check for sharp transition in all_adaptive N=4
    for comp in ["all_adaptive", "leader_followers"]:
        ds, rates = [], []
        for (c, n, d), metrics in sorted(results_by_key.items()):
            if c == comp and n == 4 and "coordination_rate" in metrics:
                ds.append(d)
                rates.append(metrics["coordination_rate"].mean)
        if ds:
            tp = detect_phase_transition(ds, rates)
            sharp = compute_sharpness(ds, rates)
            if tp is not None:
                lines.append(f"- **{comp}** (N=4): phase transition at d={tp:.3f}, sharpness={sharp:.3f}")
            else:
                lines.append(f"- **{comp}** (N=4): no phase transition detected (coordination persists or never forms)")

    report_text = "\n".join(lines)
    report_path = output_dir / "report.md"
    report_path.write_text(report_text)
    return report_text
