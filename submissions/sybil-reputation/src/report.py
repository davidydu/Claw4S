"""Generate a summary report from experiment results."""

from __future__ import annotations

import json
import os
import statistics
from collections import defaultdict
from typing import Dict, List


def _mean(values: List[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _std(values: List[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def generate_report(data: Dict) -> str:
    """Generate a Markdown report from experiment results.

    Args:
        data: The full experiment output dict (with 'metadata' and 'results').

    Returns:
        Report string in Markdown format.
    """
    meta = data["metadata"]
    results = data["results"]

    # Group results by (algorithm, strategy, n_sybil) -> list of metric dicts
    grouped: Dict[tuple, List[Dict]] = defaultdict(list)
    for r in results:
        cfg = r["config"]
        key = (cfg["algorithm"], cfg["strategy"], cfg["n_sybil"])
        grouped[key].append(r["metrics"])

    lines = []
    lines.append("# Sybil Resilience in AI Agent Reputation Networks")
    lines.append("")
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append(f"- **Honest agents:** {meta['n_honest']}")
    lines.append(f"- **Sybil counts:** {meta['sybil_counts']}")
    lines.append(f"- **Algorithms:** {', '.join(meta['algorithms'])}")
    lines.append(f"- **Strategies:** {', '.join(meta['strategies'])}")
    lines.append(f"- **Seeds:** {meta['seeds']}")
    lines.append(f"- **Rounds per sim:** {meta['n_rounds']}")
    lines.append(f"- **Total simulations:** {meta['total_simulations']}")
    lines.append(f"- **Runtime:** {meta['elapsed_seconds']}s")
    lines.append("")

    # Table 1: Reputation accuracy by algorithm and Sybil count
    lines.append("## Table 1: Reputation Accuracy (Spearman rho, honest agents)")
    lines.append("")
    sybil_counts = meta["sybil_counts"]
    algos = meta["algorithms"]

    header = "| Algorithm | " + " | ".join(f"K={k}" for k in sybil_counts) + " |"
    sep = "|---|" + "|".join("---" for _ in sybil_counts) + "|"
    lines.append(header)
    lines.append(sep)

    for algo in algos:
        row = f"| {algo} |"
        for k in sybil_counts:
            if k == 0:
                vals = [m["reputation_accuracy"] for m in grouped[(algo, "none", 0)]]
            else:
                # Average across all strategies for this K
                vals = []
                for strat in meta["strategies"]:
                    vals.extend(
                        m["reputation_accuracy"]
                        for m in grouped[(algo, strat, k)]
                    )
            row += f" {_mean(vals):.3f} +/- {_std(vals):.3f} |"
        lines.append(row)
    lines.append("")

    # Table 2: Sybil detection rate by algorithm and strategy (K=10)
    lines.append("## Table 2: Sybil Detection Rate at K=10")
    lines.append("")
    strategies = meta["strategies"]
    header2 = "| Algorithm | " + " | ".join(strategies) + " |"
    sep2 = "|---|" + "|".join("---" for _ in strategies) + "|"
    lines.append(header2)
    lines.append(sep2)

    for algo in algos:
        row = f"| {algo} |"
        for strat in strategies:
            vals = [
                m["sybil_detection_rate"] for m in grouped[(algo, strat, 10)]
            ]
            row += f" {_mean(vals):.3f} +/- {_std(vals):.3f} |"
        lines.append(row)
    lines.append("")

    # Table 3: Honest welfare by algorithm and K
    lines.append("## Table 3: Honest Agent Welfare (mean reputation)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for algo in algos:
        row = f"| {algo} |"
        for k in sybil_counts:
            if k == 0:
                vals = [m["honest_welfare"] for m in grouped[(algo, "none", 0)]]
            else:
                vals = []
                for strat in meta["strategies"]:
                    vals.extend(
                        m["honest_welfare"] for m in grouped[(algo, strat, k)]
                    )
            row += f" {_mean(vals):.3f} +/- {_std(vals):.3f} |"
        lines.append(row)
    lines.append("")

    # Table 4: Market efficiency
    lines.append("## Table 4: Market Efficiency (normalized Kendall tau)")
    lines.append("")
    lines.append(header)
    lines.append(sep)

    for algo in algos:
        row = f"| {algo} |"
        for k in sybil_counts:
            if k == 0:
                vals = [m["market_efficiency"] for m in grouped[(algo, "none", 0)]]
            else:
                vals = []
                for strat in meta["strategies"]:
                    vals.extend(
                        m["market_efficiency"] for m in grouped[(algo, strat, k)]
                    )
            row += f" {_mean(vals):.3f} +/- {_std(vals):.3f} |"
        lines.append(row)
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Find most resilient algorithm at K=20
    best_algo_k20 = None
    best_acc_k20 = -1.0
    for algo in algos:
        vals = []
        for strat in strategies:
            vals.extend(
                m["reputation_accuracy"] for m in grouped[(algo, strat, 20)]
            )
        mean_acc = _mean(vals)
        if mean_acc > best_acc_k20:
            best_acc_k20 = mean_acc
            best_algo_k20 = algo

    # Find worst strategy (most damaging at K=10)
    worst_strat = None
    worst_acc = 1.0
    for strat in strategies:
        vals = []
        for algo in algos:
            vals.extend(
                m["reputation_accuracy"] for m in grouped[(algo, strat, 10)]
            )
        mean_acc = _mean(vals)
        if mean_acc < worst_acc:
            worst_acc = mean_acc
            worst_strat = strat

    lines.append(
        f"1. **Most resilient algorithm:** {best_algo_k20} maintains "
        f"accuracy {best_acc_k20:.3f} even at K=20 Sybil agents."
    )
    lines.append(
        f"2. **Most damaging strategy:** {worst_strat} reduces mean "
        f"accuracy to {worst_acc:.3f} at K=10."
    )

    # Baseline accuracy (K=0)
    baseline_accs = []
    for algo in algos:
        for m in grouped[(algo, "none", 0)]:
            baseline_accs.append(m["reputation_accuracy"])
    lines.append(
        f"3. **Baseline accuracy (K=0):** {_mean(baseline_accs):.3f} "
        f"+/- {_std(baseline_accs):.3f} across all algorithms."
    )

    lines.append("")
    report = "\n".join(lines)

    os.makedirs("results", exist_ok=True)
    with open("results/report.md", "w") as f:
        f.write(report)
    print("Report saved to results/report.md")

    return report
