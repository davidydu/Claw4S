"""Report generation for calibration experiments.

Produces a markdown summary of key findings from the experimental results.
"""

from typing import Any


def generate_report(results: dict[str, Any]) -> str:
    """Generate a markdown report summarizing experimental findings.

    Args:
        results: Full results dict from run_all_experiments().

    Returns:
        Markdown-formatted report string.
    """
    meta = results['metadata']
    agg = results['aggregated']

    lines = [
        "# Calibration Under Distribution Shift: Results Report",
        "",
        "## Experimental Setup",
        "",
        f"- **Hidden widths:** {meta['hidden_widths']}",
        f"- **Shift magnitudes:** {meta['shift_magnitudes']}",
        f"- **Seeds:** {meta['seeds']} ({len(meta['seeds'])} runs per config)",
        f"- **Features:** {meta['n_features']}, **Classes:** {meta['n_classes']}",
        f"- **ECE bins:** {meta['n_bins']}",
        f"- **Training epochs:** {meta['train_epochs']}, **LR:** {meta['learning_rate']}",
        f"- **Total experiments:** {meta['n_experiments']}",
        f"- **Runtime:** {meta['elapsed_seconds']:.1f}s",
        "",
        "## ECE Results Table",
        "",
        "Mean ECE (std) across seeds for each (width, shift) combination:",
        "",
    ]

    # Build ECE table
    widths = sorted(set(r['hidden_width'] for r in agg))
    shifts = sorted(set(r['shift_magnitude'] for r in agg))

    header = "| Width |" + " | ".join(f"Shift={s}" for s in shifts) + " |"
    sep = "|-------|" + " | ".join("-------" for _ in shifts) + " |"
    lines.append(header)
    lines.append(sep)

    for width in widths:
        row = f"| {width:>5} |"
        for shift in shifts:
            match = [r for r in agg
                     if r['hidden_width'] == width
                     and abs(r['shift_magnitude'] - shift) < 1e-6]
            if match:
                m = match[0]
                row += f" {m['ece_mean']:.4f} ({m['ece_std']:.4f}) |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.extend(["", "## Accuracy Results Table", ""])
    header = "| Width |" + " | ".join(f"Shift={s}" for s in shifts) + " |"
    lines.append(header)
    lines.append(sep)

    for width in widths:
        row = f"| {width:>5} |"
        for shift in shifts:
            match = [r for r in agg
                     if r['hidden_width'] == width
                     and abs(r['shift_magnitude'] - shift) < 1e-6]
            if match:
                m = match[0]
                row += f" {m['accuracy_mean']:.4f} ({m['accuracy_std']:.4f}) |"
            else:
                row += " N/A |"
        lines.append(row)

    lines.extend(["", "## Brier Score Results Table", ""])
    header = "| Width |" + " | ".join(f"Shift={s}" for s in shifts) + " |"
    lines.append(header)
    lines.append(sep)

    for width in widths:
        row = f"| {width:>5} |"
        for shift in shifts:
            match = [r for r in agg
                     if r['hidden_width'] == width
                     and abs(r['shift_magnitude'] - shift) < 1e-6]
            if match:
                m = match[0]
                row += f" {m['brier_mean']:.4f} ({m['brier_std']:.4f}) |"
            else:
                row += " N/A |"
        lines.append(row)

    # Key findings
    lines.extend(["", "## Key Findings", ""])

    # Find in-distribution ECE comparison
    id_results = [r for r in agg if abs(r['shift_magnitude']) < 1e-6]
    if id_results:
        best_id = min(id_results, key=lambda r: r['ece_mean'])
        worst_id = max(id_results, key=lambda r: r['ece_mean'])
        lines.append(f"1. **In-distribution calibration:** Best ECE at width="
                     f"{best_id['hidden_width']} ({best_id['ece_mean']:.4f}), "
                     f"worst at width={worst_id['hidden_width']} "
                     f"({worst_id['ece_mean']:.4f}).")

    # Find maximum shift degradation
    max_shift = max(shifts)
    max_shift_results = [r for r in agg
                         if abs(r['shift_magnitude'] - max_shift) < 1e-6]
    if max_shift_results and id_results:
        for width in widths:
            id_match = [r for r in id_results if r['hidden_width'] == width]
            shift_match = [r for r in max_shift_results
                          if r['hidden_width'] == width]
            if id_match and shift_match:
                degradation = shift_match[0]['ece_mean'] - id_match[0]['ece_mean']
                lines.append(f"   - Width {width}: ECE degrades by "
                             f"{degradation:+.4f} from shift=0 to shift={max_shift}")

    # Overconfidence analysis
    lines.extend(["", "2. **Overconfidence under shift:**"])
    for width in widths:
        max_s = [r for r in max_shift_results if r['hidden_width'] == width]
        if max_s:
            gap = max_s[0]['confidence_mean'] - max_s[0]['accuracy_mean']
            lines.append(f"   - Width {width}: overconfidence gap = "
                         f"{gap:.4f} at shift={max_shift}")

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Synthetic Gaussian cluster data may not reflect real-world shift patterns.",
        "- Only covariate shift (mean translation) is tested; other shift types "
        "(e.g., label shift, concept drift) are not covered.",
        "- 2-layer MLPs are a simplified architecture; deeper networks or "
        "transformers may show different calibration behavior.",
        "- Small sample sizes (500 train, 200 test) may introduce variance.",
        "",
    ])

    return "\n".join(lines)
