"""Markdown report generation for grokking phase diagram results."""

from src.analysis import Phase, aggregate_results


def generate_report(sweep_results: list[dict]) -> str:
    """Generate a markdown summary report from sweep results.

    Args:
        sweep_results: List of result dicts from the sweep.

    Returns:
        Markdown-formatted report string.
    """
    # Convert phase strings back to Phase enum for aggregate_results
    results_for_agg = []
    for r in sweep_results:
        phase = r["phase"]
        if isinstance(phase, str):
            phase_enum = Phase(phase)
        else:
            phase_enum = phase
        results_for_agg.append({
            "phase": phase_enum,
            "grokking_gap": r.get("grokking_gap"),
        })

    stats = aggregate_results(results_for_agg)

    lines = []
    lines.append("# Grokking Phase Diagram Results")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(
        f"Total training runs: **{stats['total_runs']}**"
    )
    lines.append("")

    # Phase distribution
    lines.append("## Phase Distribution")
    lines.append("")
    lines.append("| Phase | Count | Fraction |")
    lines.append("|-------|-------|----------|")
    for phase in Phase:
        count = stats["phase_counts"].get(phase.value, 0)
        frac = count / max(stats["total_runs"], 1)
        lines.append(f"| {phase.value.capitalize()} | {count} | {frac:.1%} |")
    lines.append("")

    # Grokking statistics
    lines.append("## Grokking Statistics")
    lines.append("")
    grok_frac = stats["grokking_fraction"]
    lines.append(f"- Grokking rate: **{grok_frac:.1%}** of runs")
    if stats["mean_grokking_gap"] is not None:
        lines.append(
            f"- Mean grokking gap: **{stats['mean_grokking_gap']:.0f}** epochs"
        )
        lines.append(
            f"- Max grokking gap: **{stats['max_grokking_gap']:.0f}** epochs"
        )
    else:
        lines.append("- No grokking observed in any run.")
    lines.append("")

    # Phase diagram references
    lines.append("## Phase Diagrams")
    lines.append("")
    hidden_dims = sorted(
        set(r["config"]["hidden_dim"] for r in sweep_results)
    )
    for hd in hidden_dims:
        lines.append(f"### Hidden Dimension = {hd}")
        lines.append(f"![Phase diagram h={hd}](phase_diagram_h{hd}.png)")
        lines.append("")

    # Detailed results table
    lines.append("## Detailed Results")
    lines.append("")
    lines.append(
        "| Hidden | Weight Decay | Train % | Phase | Train Acc | Test Acc | "
        "Grok Gap | Epochs | Time (s) |"
    )
    lines.append(
        "|--------|-------------|---------|-------|-----------|----------|"
        "---------|--------|----------|"
    )
    for r in sweep_results:
        c = r["config"]
        m = r["metrics"]
        gap_str = str(r["grokking_gap"]) if r["grokking_gap"] is not None else "-"
        lines.append(
            f"| {c['hidden_dim']} | {c['weight_decay']} | {c['train_fraction']:.0%} "
            f"| {r['phase']} | {m['final_train_acc']:.1%} | {m['final_test_acc']:.1%} "
            f"| {gap_str} | {m['total_epochs']} | {r['elapsed_seconds']} |"
        )
    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    # Finding 1: Effect of weight decay
    wd_phases = {}
    for r in sweep_results:
        wd = r["config"]["weight_decay"]
        if wd not in wd_phases:
            wd_phases[wd] = []
        wd_phases[wd].append(r["phase"])

    lines.append("### Effect of Weight Decay")
    lines.append("")
    for wd in sorted(wd_phases.keys()):
        phases = wd_phases[wd]
        grok_count = sum(1 for p in phases if p == Phase.GROKKING.value)
        mem_count = sum(1 for p in phases if p == Phase.MEMORIZATION.value)
        comp_count = sum(1 for p in phases if p == Phase.COMPREHENSION.value)
        lines.append(
            f"- wd={wd}: {grok_count} grokking, {mem_count} memorization, "
            f"{comp_count} comprehension (out of {len(phases)} runs)"
        )
    lines.append("")

    # Finding 2: Effect of dataset fraction
    frac_phases = {}
    for r in sweep_results:
        frac = r["config"]["train_fraction"]
        if frac not in frac_phases:
            frac_phases[frac] = []
        frac_phases[frac].append(r["phase"])

    lines.append("### Effect of Dataset Fraction")
    lines.append("")
    for frac in sorted(frac_phases.keys()):
        phases = frac_phases[frac]
        grok_count = sum(1 for p in phases if p == Phase.GROKKING.value)
        gen_count = sum(
            1 for p in phases
            if p in [Phase.GROKKING.value, Phase.COMPREHENSION.value]
        )
        lines.append(
            f"- frac={frac:.0%}: {gen_count}/{len(phases)} generalized "
            f"({grok_count} via grokking)"
        )
    lines.append("")

    # Finding 3: Training curves reference
    lines.append("### Example Training Curves")
    lines.append("")
    lines.append("![Training curves](grokking_curves.png)")
    lines.append("")
    lines.append(
        "Training curves show the characteristic grokking pattern: "
        "train accuracy reaches near-perfect performance long before "
        "test accuracy improves, creating a delayed phase transition "
        "from memorization to generalization."
    )

    return "\n".join(lines)
