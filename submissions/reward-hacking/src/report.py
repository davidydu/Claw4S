"""Report generator: summarizes experiment results as markdown."""

from __future__ import annotations

import json
import os
from pathlib import Path
from itertools import groupby
from operator import itemgetter

import numpy as np

from src.metrics import aggregate_across_seeds


def generate_report(results: list[dict], output_dir: str = "results") -> str:
    """Generate a markdown report from experiment results.

    Also saves results.json and report.md to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save raw results
    results_path = Path(output_dir) / "results.json"
    with open(results_path, "w") as f:
        json.dump({
            "metadata": {
                "n_simulations": len(results),
                "n_agents": 10,
                "n_rounds": 5000,
                "experiment": "reward-hacking-propagation",
            },
            "results": results,
        }, f, indent=2, default=str)
    print(f"Saved raw results to {results_path}")

    # Build report
    lines: list[str] = []
    lines.append("# Reward Hacking Propagation: Experiment Report\n")
    lines.append(f"Total simulations: {len(results)}\n")

    # --- Table 1: Adoption by topology x monitor fraction ---
    lines.append("## 1. Steady-State Adoption Rate by Topology and Monitor Fraction\n")
    lines.append("| Topology | Monitors 0% | Monitors 10% | Monitors 25% | Monitors 50% |")
    lines.append("|----------|-------------|--------------|--------------|--------------|")

    for topo in ["grid", "random", "star"]:
        row = f"| {topo:8s} "
        for mf in [0.0, 0.1, 0.25, 0.5]:
            subset = [r["metrics"] for r in results
                      if r["params"]["topology"] == topo
                      and r["params"]["monitor_fraction"] == mf]
            agg = aggregate_across_seeds(subset)
            mean_val = agg.get("steady_state_adoption_mean", 0.0)
            std_val = agg.get("steady_state_adoption_std", 0.0)
            row += f"| {mean_val:.2f} +/- {std_val:.2f} "
        row += "|"
        lines.append(row)

    # --- Table 2: Propagation speed ---
    lines.append("\n## 2. Propagation Speed (Rounds to 50% Adoption)\n")
    lines.append("| Initial Hackers | Grid | Random | Star |")
    lines.append("|-----------------|------|--------|------|")

    for nh in [1, 2, 5]:
        row = f"| {nh:15d} "
        for topo in ["grid", "random", "star"]:
            subset = [r["metrics"] for r in results
                      if r["params"]["n_initial_hackers"] == nh
                      and r["params"]["topology"] == topo
                      and r["params"]["monitor_fraction"] == 0.0]
            agg = aggregate_across_seeds(subset)
            t50 = agg.get("time_to_50pct_mean")
            frac = agg.get("time_to_50pct_reached_frac", 0.0)
            if t50 is not None:
                row += f"| {t50:.0f} ({frac*100:.0f}%) "
            else:
                row += f"| N/A ({frac*100:.0f}%) "
        row += "|"
        lines.append(row)

    # --- Table 3: Containment effectiveness ---
    lines.append("\n## 3. Containment Effectiveness\n")
    lines.append("| Detectability | Monitors 10% | Monitors 25% | Monitors 50% |")
    lines.append("|---------------|--------------|--------------|--------------|")

    for det in ["obvious", "subtle", "invisible"]:
        row = f"| {det:13s} "
        for mf in [0.1, 0.25, 0.5]:
            subset = [r["metrics"] for r in results
                      if r["params"]["detectability"] == det
                      and r["params"]["monitor_fraction"] == mf]
            agg = aggregate_across_seeds(subset)
            cr = agg.get("containment_rate", 0.0)
            row += f"| {cr*100:.0f}% "
        row += "|"
        lines.append(row)

    # --- Table 4: Welfare impact ---
    lines.append("\n## 4. Welfare Impact (Proxy-True Reward Divergence)\n")
    lines.append("| Topology | Monitor 0% | Monitor 25% | Monitor 50% |")
    lines.append("|----------|------------|-------------|-------------|")

    for topo in ["grid", "random", "star"]:
        row = f"| {topo:8s} "
        for mf in [0.0, 0.25, 0.5]:
            subset = [r["metrics"] for r in results
                      if r["params"]["topology"] == topo
                      and r["params"]["monitor_fraction"] == mf]
            agg = aggregate_across_seeds(subset)
            div_mean = agg.get("steady_state_divergence_mean", 0.0)
            div_std = agg.get("steady_state_divergence_std", 0.0)
            row += f"| {div_mean:.3f} +/- {div_std:.3f} "
        row += "|"
        lines.append(row)

    # --- Key findings ---
    lines.append("\n## 5. Key Findings\n")

    # Find highest-spread condition (no monitors)
    no_monitor = [r for r in results if r["params"]["monitor_fraction"] == 0.0]
    if no_monitor:
        worst = max(no_monitor, key=lambda r: r["metrics"]["steady_state_adoption"])
        lines.append(
            f"- **Worst case (no monitors):** {worst['params']['topology']} topology, "
            f"{worst['params']['n_initial_hackers']} initial hackers -> "
            f"{worst['metrics']['steady_state_adoption']:.0%} steady-state adoption"
        )

    # Find best containment condition
    with_monitor = [r for r in results if r["params"]["monitor_fraction"] > 0]
    if with_monitor:
        best = min(with_monitor, key=lambda r: r["metrics"]["steady_state_adoption"])
        lines.append(
            f"- **Best containment:** {best['params']['topology']} topology, "
            f"{best['params']['monitor_fraction']:.0%} monitors, "
            f"{best['params']['detectability']} detectability -> "
            f"{best['metrics']['steady_state_adoption']:.0%} steady-state adoption"
        )

    report = "\n".join(lines) + "\n"

    report_path = Path(output_dir) / "report.md"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved report to {report_path}")

    return report
