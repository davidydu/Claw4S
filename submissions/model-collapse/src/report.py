"""Generate a human-readable Markdown report from experiment results."""

from __future__ import annotations

from typing import Any

from .analysis import aggregate_by_condition, anchor_effectiveness, build_summary


def generate_report(results: list[dict[str, Any]]) -> str:
    """Produce a Markdown report summarising the experiment."""
    aggregated = aggregate_by_condition(results)
    summary = build_summary(aggregated)
    anchor = anchor_effectiveness(aggregated)

    lines: list[str] = []
    lines.append("# Model Collapse Experiment Report\n")

    # ---- Overview -----------------------------------------------------------
    n_sims = len(results)
    agent_types = sorted({r["config"]["agent_type"] for r in results})
    gt_fracs = sorted({r["config"]["gt_fraction"] for r in results})
    dists = sorted({r["config"]["dist_name"] for r in results})
    n_gen = results[0]["config"]["n_generations"]

    lines.append("## Overview\n")
    lines.append(f"- **Simulations:** {n_sims}")
    lines.append(f"- **Agent types:** {', '.join(agent_types)}")
    lines.append(f"- **GT fractions:** {', '.join(f'{g:.0%}' for g in gt_fracs)}")
    lines.append(f"- **Distributions:** {', '.join(dists)}")
    lines.append(f"- **Generations:** {n_gen}")
    lines.append("")

    # ---- Summary table ------------------------------------------------------
    lines.append("## Summary Table\n")
    lines.append(
        "| Agent | GT% | Distribution | Final KL (mean +/- std) | "
        "Collapse Gen | Curve Shape |"
    )
    lines.append("|-------|-----|--------------|-------------------------|" "-------------|-------------|")
    for r in summary:
        lines.append(
            f"| {r['agent_type']} | {r['gt_fraction']:.0%} | {r['dist_name']} | "
            f"{r['final_kl_mean']:.3f} +/- {r['final_kl_std']:.3f} | "
            f"{r['mean_collapse_gen']:.1f} | {r['curve_shape']} |"
        )
    lines.append("")

    # ---- Collapse dynamics --------------------------------------------------
    lines.append("## Collapse Dynamics\n")
    lines.append("### KL Divergence Trajectories (averaged over seeds)\n")

    for at in agent_types:
        lines.append(f"#### {at.title()} Agent\n")
        lines.append("| Gen | " + " | ".join(f"GT={g:.0%}" for g in gt_fracs) + " |")
        lines.append("|-----|" + "|".join(["-------"] * len(gt_fracs)) + "|")
        # Average across distributions for the trajectory table
        for gen_i in range(n_gen):
            vals = []
            for gf in gt_fracs:
                kls = [
                    agg["mean_kl"][gen_i]
                    for (a, g, d), agg in aggregated.items()
                    if a == at and g == gf
                ]
                avg = sum(kls) / len(kls) if kls else float("nan")
                vals.append(f"{avg:.4f}")
            lines.append(f"| {gen_i} | " + " | ".join(vals) + " |")
        lines.append("")

    # ---- Anchor effectiveness -----------------------------------------------
    lines.append("## Anchor Effectiveness\n")
    for at in agent_types:
        if at not in anchor:
            continue
        lines.append(f"### {at.title()} Agent\n")
        lines.append("| Distribution | GT% | Mean Collapse Gen | Delay/1% GT |")
        lines.append("|--------------|-----|-------------------|-------------|")
        for e in anchor[at]:
            d_str = f"{e['delta_per_pct']:.2f}" if e["delta_per_pct"] is not None else "baseline"
            lines.append(
                f"| {e['dist_name']} | {e['gt_fraction']:.0%} | "
                f"{e['mean_collapse']:.1f} | {d_str} |"
            )
        lines.append("")

    # ---- Key findings -------------------------------------------------------
    lines.append("## Key Findings\n")

    # Find collapse rate for naive agents at 0% GT
    naive_zero = [
        s for s in summary
        if s["agent_type"] == "naive" and s["gt_fraction"] == 0.0
    ]
    if naive_zero:
        avg_collapse = sum(s["mean_collapse_gen"] for s in naive_zero) / len(naive_zero)
        lines.append(
            f"1. **Naive agents without ground truth** collapse at generation "
            f"{avg_collapse:.1f} on average (KL > 1.0 nats)."
        )

    # Find minimum GT fraction that prevents collapse for anchored
    anchored_stable = [
        s for s in summary
        if s["agent_type"] == "anchored" and s["mean_collapse_gen"] >= n_gen
    ]
    if anchored_stable:
        min_gt = min(s["gt_fraction"] for s in anchored_stable)
        lines.append(
            f"2. **Anchored agents** remain stable with as little as "
            f"{min_gt:.0%} ground truth per generation."
        )

    # Compare selective vs naive
    sel_zero = [
        s for s in summary
        if s["agent_type"] == "selective" and s["gt_fraction"] == 0.0
    ]
    if sel_zero and naive_zero:
        sel_avg = sum(s["mean_collapse_gen"] for s in sel_zero) / len(sel_zero)
        nav_avg = sum(s["mean_collapse_gen"] for s in naive_zero) / len(naive_zero)
        diff = sel_avg - nav_avg
        lines.append(
            f"3. **Selective filtering** delays collapse by "
            f"{diff:+.1f} generations compared to naive agents."
        )

    # Exponential vs linear
    exp_count = sum(1 for s in summary if s["curve_shape"] == "exponential")
    lin_count = sum(1 for s in summary if s["curve_shape"] == "linear")
    sta_count = sum(1 for s in summary if s["curve_shape"] == "stable")
    lines.append(
        f"4. **Curve shapes:** {exp_count} exponential, {lin_count} linear, "
        f"{sta_count} stable out of {len(summary)} conditions."
    )

    lines.append("")
    return "\n".join(lines)
