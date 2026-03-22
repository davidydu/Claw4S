"""Generate a Markdown report from experiment results."""

from __future__ import annotations

import json
import os


def generate_report(results: dict, output_dir: str = "results") -> str:
    """Generate a human-readable Markdown report from aggregated results."""
    agg = results["aggregated"]
    meta = results["metadata"]

    lines = [
        "# Delegation Game: Experiment Report",
        "",
        f"**Simulations:** {meta['num_simulations']} "
        f"({meta['num_rounds_per_sim']} rounds each)",
        f"**Schemes:** {', '.join(meta['schemes'])}",
        f"**Seeds:** {meta['seeds']}",
        f"**Runtime:** {meta['elapsed_seconds']}s",
        "",
        "## Summary: Average Quality by Scheme and Worker Composition",
        "",
    ]

    # Build summary table: schemes as rows, compositions as columns
    compositions = list(meta["worker_compositions"].keys())
    noise_levels = list(meta["noise_levels"].keys())

    for noise_name in noise_levels:
        noise_val = meta["noise_levels"][noise_name]
        lines.append(f"### Noise: {noise_name} (std={noise_val})")
        lines.append("")

        # Table header
        header = "| Scheme | " + " | ".join(compositions) + " |"
        sep = "|--------|" + "|".join(["--------"] * len(compositions)) + "|"
        lines.append(header)
        lines.append(sep)

        for scheme in meta["schemes"]:
            row = f"| {scheme} |"
            for comp in compositions:
                wtypes = meta["worker_compositions"][comp]
                match = [
                    a for a in agg
                    if a["scheme"] == scheme
                    and sorted(a["worker_types"]) == sorted(wtypes)
                    and a["noise_std"] == noise_val
                ]
                if match:
                    m = match[0]
                    row += (f" {m['avg_quality_mean']:.2f} "
                            f"(+/-{m['avg_quality_std']:.2f}) |")
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

    # Incentive efficiency table
    lines.append("## Incentive Efficiency (Quality per Dollar)")
    lines.append("")
    for noise_name in noise_levels:
        noise_val = meta["noise_levels"][noise_name]
        lines.append(f"### Noise: {noise_name} (std={noise_val})")
        lines.append("")
        header = "| Scheme | " + " | ".join(compositions) + " |"
        sep = "|--------|" + "|".join(["--------"] * len(compositions)) + "|"
        lines.append(header)
        lines.append(sep)

        for scheme in meta["schemes"]:
            row = f"| {scheme} |"
            for comp in compositions:
                wtypes = meta["worker_compositions"][comp]
                match = [
                    a for a in agg
                    if a["scheme"] == scheme
                    and sorted(a["worker_types"]) == sorted(wtypes)
                    and a["noise_std"] == noise_val
                ]
                if match:
                    m = match[0]
                    row += (f" {m['incentive_efficiency_mean']:.2f} "
                            f"(+/-{m['incentive_efficiency_std']:.2f}) |")
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

    # Shirking rate table
    lines.append("## Shirking Rate (fraction effort < 3)")
    lines.append("")
    for noise_name in noise_levels:
        noise_val = meta["noise_levels"][noise_name]
        lines.append(f"### Noise: {noise_name} (std={noise_val})")
        lines.append("")
        header = "| Scheme | " + " | ".join(compositions) + " |"
        sep = "|--------|" + "|".join(["--------"] * len(compositions)) + "|"
        lines.append(header)
        lines.append(sep)

        for scheme in meta["schemes"]:
            row = f"| {scheme} |"
            for comp in compositions:
                wtypes = meta["worker_compositions"][comp]
                match = [
                    a for a in agg
                    if a["scheme"] == scheme
                    and sorted(a["worker_types"]) == sorted(wtypes)
                    and a["noise_std"] == noise_val
                ]
                if match:
                    m = match[0]
                    row += (f" {m['shirking_rate_mean']:.2f} "
                            f"(+/-{m['shirking_rate_std']:.2f}) |")
                else:
                    row += " - |"
            lines.append(row)
        lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")
    lines.extend(_extract_findings(agg, meta))

    report = "\n".join(lines)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, "report.md")
    with open(report_path, "w") as f:
        f.write(report)

    return report


def _extract_findings(agg: list[dict], meta: dict) -> list[str]:
    """Extract key findings from aggregated results."""
    findings = []

    # Best scheme per noise level
    for noise_name, noise_val in meta["noise_levels"].items():
        entries = [a for a in agg if a["noise_std"] == noise_val]
        if not entries:
            continue
        best = max(entries, key=lambda x: x["incentive_efficiency_mean"])
        findings.append(
            f"- **{noise_name.capitalize()} noise:** Best incentive efficiency = "
            f"{best['scheme']} with {'-'.join(sorted(best['worker_types']))} "
            f"(efficiency={best['incentive_efficiency_mean']:.2f})"
        )

    findings.append("")

    # Shirking analysis
    for scheme in meta["schemes"]:
        scheme_entries = [a for a in agg if a["scheme"] == scheme]
        if not scheme_entries:
            continue
        avg_shirk = sum(
            e["shirking_rate_mean"] for e in scheme_entries
        ) / len(scheme_entries)
        findings.append(
            f"- **{scheme}:** Average shirking rate = {avg_shirk:.2%}"
        )

    findings.append("")

    # Robustness: which scheme has lowest variance in quality across conditions
    scheme_variances = {}
    for scheme in meta["schemes"]:
        entries = [a for a in agg if a["scheme"] == scheme]
        if entries:
            import numpy as np
            quals = [e["avg_quality_mean"] for e in entries]
            scheme_variances[scheme] = float(np.std(quals))

    if scheme_variances:
        most_robust = min(scheme_variances, key=scheme_variances.get)
        findings.append(
            f"- **Most robust scheme:** {most_robust} "
            f"(lowest quality std across conditions = "
            f"{scheme_variances[most_robust]:.2f})"
        )

    return findings
