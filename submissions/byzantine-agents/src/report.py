"""Generate a Markdown summary report from experiment results."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def generate_report(results: dict[str, Any]) -> str:
    """Produce a human-readable Markdown report."""
    meta = results["metadata"]
    lines = [
        "# Byzantine Fault Tolerance in Multi-Agent Decision Systems",
        "",
        "## Experiment Summary",
        "",
        f"- **Configurations:** {meta['total_configs']}",
        f"- **Honest types:** {', '.join(meta['honest_types'])}",
        f"- **Byzantine types:** {', '.join(meta['byzantine_types'])}",
        f"- **Byzantine fractions:** {meta['fractions']}",
        f"- **Committee sizes:** {meta['committee_sizes']}",
        f"- **Seeds:** {meta['seeds']}",
        f"- **Rounds per sim:** {meta['rounds_per_sim']}",
        f"- **Workers:** {meta['n_workers']}",
        f"- **Elapsed:** {meta['elapsed_seconds']:.1f}s",
        "",
        "## Key Findings",
        "",
    ]

    # Derived metrics table
    lines.append("### Byzantine Thresholds (accuracy < 50%)")
    lines.append("")
    lines.append("| Honest Type | Byzantine Type | N | Threshold f* | Resilience |")
    lines.append("|-------------|---------------|---|-------------|------------|")
    for d in sorted(results["derived_metrics"], key=lambda x: (x["honest_type"], x["byzantine_type"], x["committee_size"])):
        lines.append(
            f"| {d['honest_type']} | {d['byzantine_type']} | {d['committee_size']} "
            f"| {d['byzantine_threshold_50']:.2f} | {d['resilience_score']:.3f} |"
        )
    lines.append("")

    # Amplification table
    if results.get("amplifications"):
        lines.append("### Byzantine Amplification (Strategic vs Random at f=0.33)")
        lines.append("")
        lines.append("| Honest Type | N | Amplification | Baseline Acc | Strategic Acc | Random Acc |")
        lines.append("|-------------|---|--------------|-------------|--------------|------------|")
        for a in sorted(results["amplifications"], key=lambda x: (x["honest_type"], x["committee_size"])):
            lines.append(
                f"| {a['honest_type']} | {a['committee_size']} "
                f"| {a['amplification_at_f33']:.2f}x "
                f"| {a['baseline_accuracy']:.3f} "
                f"| {a['strategic_accuracy_f33']:.3f} "
                f"| {a['random_accuracy_f33']:.3f} |"
            )
        lines.append("")

    # Accuracy summary per honest type across fractions (averaged over byz types and sizes)
    lines.append("### Mean Accuracy by Honest Type and Byzantine Fraction")
    lines.append("")
    lines.append("| Honest Type | f=0.00 | f=0.10 | f=0.20 | f=0.33 | f=0.50 |")
    lines.append("|-------------|--------|--------|--------|--------|--------|")

    from collections import defaultdict
    acc_by_honest_frac: dict[tuple[str, float], list[float]] = defaultdict(list)
    for s in results["summaries"]:
        acc_by_honest_frac[(s["honest_type"], s["byzantine_fraction"])].append(s["mean_accuracy"])

    for honest in meta["honest_types"]:
        cells = []
        for f in meta["fractions"]:
            vals = acc_by_honest_frac.get((honest, round(f, 4)), [])
            if vals:
                import numpy as np
                cells.append(f"{float(np.mean(vals)):.3f}")
            else:
                cells.append("-")
        lines.append(f"| {honest} | {' | '.join(cells)} |")
    lines.append("")

    return "\n".join(lines)


def save_report(report: str, out_dir: str = "results") -> Path:
    """Save the report to a Markdown file."""
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    out_file = path / "report.md"
    out_file.write_text(report)
    return out_file
