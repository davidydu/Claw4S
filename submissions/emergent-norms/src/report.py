"""Generate a summary report from experiment results."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def generate_report(results: list[dict[str, Any]]) -> str:
    """Generate a human-readable summary report.

    Parameters
    ----------
    results : list[dict]
        Output from run_experiment().

    Returns
    -------
    str
        Markdown-formatted report.
    """
    lines: list[str] = []
    lines.append("# Emergent Social Norms — Experiment Report")
    lines.append("")
    lines.append(f"Total simulations: {len(results)}")
    lines.append("")

    # --- Aggregate by composition ---
    lines.append("## 1. Convergence by Population Composition")
    lines.append("")
    lines.append("| Composition | Avg Convergence | Converged (%) | Avg Efficiency |")
    lines.append("|-------------|-----------------|---------------|----------------|")

    by_comp: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_comp[r["composition_name"]].append(r)

    for comp_name, group in sorted(by_comp.items()):
        conv_times = [r["convergence_time"] for r in group]
        total_rounds = group[0]["total_rounds"]
        converged_frac = sum(1 for c in conv_times if c < total_rounds) / len(conv_times)
        avg_conv = np.mean(conv_times)
        avg_eff = np.mean([r["efficiency"] for r in group])
        lines.append(
            f"| {comp_name} | {avg_conv:.0f} | {converged_frac*100:.1f} | {avg_eff:.3f} |"
        )

    # --- Aggregate by game structure ---
    lines.append("")
    lines.append("## 2. Norm Efficiency by Game Structure")
    lines.append("")
    lines.append("| Game | Avg Efficiency | Avg Diversity | Avg Fragility |")
    lines.append("|------|----------------|---------------|---------------|")

    by_game: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_game[r["game"]].append(r)

    for game_name, group in sorted(by_game.items()):
        avg_eff = np.mean([r["efficiency"] for r in group])
        avg_div = np.mean([r["diversity"] for r in group])
        avg_frag = np.mean([r["fragility"] for r in group])
        lines.append(
            f"| {game_name} | {avg_eff:.3f} | {avg_div:.2f} | {avg_frag:.2f} |"
        )

    # --- Aggregate by population size ---
    lines.append("")
    lines.append("## 3. Scale Effects")
    lines.append("")
    lines.append("| Pop Size | Avg Convergence | Avg Efficiency | Avg Diversity |")
    lines.append("|----------|-----------------|----------------|---------------|")

    by_size: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_size[r["population_size"]].append(r)

    for size, group in sorted(by_size.items()):
        conv_times = [r["convergence_time"] for r in group]
        avg_conv = np.mean(conv_times)
        avg_eff = np.mean([r["efficiency"] for r in group])
        avg_div = np.mean([r["diversity"] for r in group])
        lines.append(
            f"| {size} | {avg_conv:.0f} | {avg_eff:.3f} | {avg_div:.2f} |"
        )

    # --- Key findings ---
    lines.append("")
    lines.append("## 4. Key Findings")
    lines.append("")

    all_eff = [r["efficiency"] for r in results]
    all_conv = [r["convergence_time"] for r in results]
    total_rounds = results[0]["total_rounds"]
    converged = sum(1 for c in all_conv if c < total_rounds)

    lines.append(f"- **Overall convergence rate:** {converged}/{len(results)} "
                 f"({converged/len(results)*100:.1f}%) of simulations converged "
                 f"within {total_rounds:,} rounds.")
    lines.append(f"- **Average norm efficiency:** {np.mean(all_eff):.3f} "
                 f"(std: {np.std(all_eff):.3f})")
    lines.append(f"- **Average norm diversity:** "
                 f"{np.mean([r['diversity'] for r in results]):.2f} clusters")

    # Best and worst compositions
    comp_eff = {
        comp: np.mean([r["efficiency"] for r in group])
        for comp, group in by_comp.items()
    }
    best_comp = max(comp_eff, key=comp_eff.get)  # type: ignore[arg-type]
    worst_comp = min(comp_eff, key=comp_eff.get)  # type: ignore[arg-type]
    lines.append(f"- **Most efficient composition:** {best_comp} "
                 f"(avg efficiency {comp_eff[best_comp]:.3f})")
    lines.append(f"- **Least efficient composition:** {worst_comp} "
                 f"(avg efficiency {comp_eff[worst_comp]:.3f})")

    # Fragility insight
    avg_frag = np.mean([r["fragility"] for r in results])
    lines.append(f"- **Average norm fragility:** {avg_frag:.2f} "
                 f"(fraction of innovators to break norm)")

    lines.append("")
    return "\n".join(lines)
