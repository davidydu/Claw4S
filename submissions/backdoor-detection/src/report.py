"""Generate summary report and visualizations from experiment results."""

import json
import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_report(results: list[dict], output_dir: str = "results") -> str:
    """Generate a markdown summary report from experiment results.

    Args:
        results: List of experiment result dicts.
        output_dir: Directory to write report and figures.

    Returns:
        Report text as a string.
    """
    os.makedirs(output_dir, exist_ok=True)

    lines = []
    lines.append("# Backdoor Detection via Spectral Signatures: Results")
    lines.append("")
    lines.append(f"Total experiments: {len(results)}")
    lines.append("")

    # Summary statistics
    aucs = [r["detection_auc"] for r in results]
    lines.append(f"Detection AUC: mean={np.mean(aucs):.3f}, "
                 f"std={np.std(aucs):.3f}, "
                 f"min={np.min(aucs):.3f}, max={np.max(aucs):.3f}")
    lines.append("")

    # Key finding: AUC by poison fraction
    lines.append("## Detection AUC by Poison Fraction")
    lines.append("")
    lines.append("| Poison % | Mean AUC | Std AUC | Min AUC | Max AUC |")
    lines.append("|----------|----------|---------|---------|---------|")

    poison_fracs = sorted(set(r["config"]["poison_fraction"] for r in results))
    for pf in poison_fracs:
        subset = [r for r in results if r["config"]["poison_fraction"] == pf]
        a = [r["detection_auc"] for r in subset]
        lines.append(f"| {pf*100:.0f}% | {np.mean(a):.3f} | {np.std(a):.3f} | "
                     f"{np.min(a):.3f} | {np.max(a):.3f} |")

    lines.append("")

    # AUC by trigger strength
    lines.append("## Detection AUC by Trigger Strength")
    lines.append("")
    lines.append("| Strength | Mean AUC | Std AUC |")
    lines.append("|----------|----------|---------|")

    strengths = sorted(set(r["config"]["trigger_strength"] for r in results))
    for ts in strengths:
        subset = [r for r in results if r["config"]["trigger_strength"] == ts]
        a = [r["detection_auc"] for r in subset]
        lines.append(f"| {ts} | {np.mean(a):.3f} | {np.std(a):.3f} |")

    lines.append("")

    # AUC by hidden dim
    lines.append("## Detection AUC by Model Size (Hidden Dim)")
    lines.append("")
    lines.append("| Hidden Dim | Mean AUC | Std AUC |")
    lines.append("|------------|----------|---------|")

    dims = sorted(set(r["config"]["hidden_dim"] for r in results))
    for hd in dims:
        subset = [r for r in results if r["config"]["hidden_dim"] == hd]
        a = [r["detection_auc"] for r in subset]
        lines.append(f"| {hd} | {np.mean(a):.3f} | {np.std(a):.3f} |")

    lines.append("")

    # Backdoor success rate
    lines.append("## Backdoor Attack Success Rate")
    lines.append("")
    sr = [r["backdoor_success_rate"] for r in results]
    lines.append(f"Mean backdoor success rate: {np.mean(sr):.3f}")
    lines.append(f"Model accuracy (clean): mean={np.mean([r['clean_model_accuracy'] for r in results]):.3f}")
    lines.append(f"Model accuracy (backdoored, on clean data): "
                 f"mean={np.mean([r['backdoored_model_accuracy'] for r in results]):.3f}")
    lines.append("")

    # Eigenvalue ratio analysis
    lines.append("## Eigenvalue Ratio (Top / Second)")
    lines.append("")
    for pf in poison_fracs:
        subset = [r for r in results if r["config"]["poison_fraction"] == pf]
        ratios = [r["eigenvalue_ratio"] for r in subset]
        lines.append(f"Poison {pf*100:.0f}%: mean ratio = {np.mean(ratios):.2f}")

    lines.append("")

    # Key findings
    lines.append("## Key Findings")
    lines.append("")

    high_auc = [r for r in results if r["detection_auc"] >= 0.9]
    lines.append(f"- {len(high_auc)}/{len(results)} experiments achieved AUC >= 0.9")

    if poison_fracs:
        best_pf_group = max(poison_fracs, key=lambda pf: np.mean(
            [r["detection_auc"] for r in results if r["config"]["poison_fraction"] == pf]))
        lines.append(f"- Best detection at poison fraction = {best_pf_group*100:.0f}%")

    if strengths:
        best_ts_group = max(strengths, key=lambda ts: np.mean(
            [r["detection_auc"] for r in results if r["config"]["trigger_strength"] == ts]))
        lines.append(f"- Strongest detection at trigger strength = {best_ts_group}")

    report_text = "\n".join(lines)

    # Save report
    with open(os.path.join(output_dir, "report.md"), "w") as f:
        f.write(report_text)

    return report_text


def generate_figures(results: list[dict], output_dir: str = "results") -> list[str]:
    """Generate publication-quality figures from experiment results.

    Creates:
    1. Heatmap: AUC vs poison fraction and trigger strength
    2. Line plot: AUC vs poison fraction by model size
    3. Bar chart: eigenvalue ratios by poison fraction

    Args:
        results: List of experiment result dicts.
        output_dir: Directory to write figures.

    Returns:
        List of figure file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    figure_paths = []

    poison_fracs = sorted(set(r["config"]["poison_fraction"] for r in results))
    strengths = sorted(set(r["config"]["trigger_strength"] for r in results))
    dims = sorted(set(r["config"]["hidden_dim"] for r in results))

    # Figure 1: Heatmap of AUC (poison fraction vs trigger strength, averaged over hidden dims)
    fig, ax = plt.subplots(figsize=(7, 5))
    auc_matrix = np.zeros((len(poison_fracs), len(strengths)))
    for i, pf in enumerate(poison_fracs):
        for j, ts in enumerate(strengths):
            subset = [r for r in results
                      if r["config"]["poison_fraction"] == pf
                      and r["config"]["trigger_strength"] == ts]
            auc_matrix[i, j] = np.mean([r["detection_auc"] for r in subset])

    im = ax.imshow(auc_matrix, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(strengths)))
    ax.set_xticklabels([str(s) for s in strengths])
    ax.set_yticks(range(len(poison_fracs)))
    ax.set_yticklabels([f"{pf*100:.0f}%" for pf in poison_fracs])
    ax.set_xlabel("Trigger Strength")
    ax.set_ylabel("Poison Fraction")
    ax.set_title("Detection AUC: Poison Fraction vs Trigger Strength")

    for i in range(len(poison_fracs)):
        for j in range(len(strengths)):
            ax.text(j, i, f"{auc_matrix[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=11, fontweight="bold")

    plt.colorbar(im, ax=ax, label="AUC")
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_auc_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    figure_paths.append(path)

    # Figure 2: Line plot of AUC vs poison fraction by model size
    fig, ax = plt.subplots(figsize=(7, 5))
    for hd in dims:
        aucs_by_pf = []
        for pf in poison_fracs:
            subset = [r for r in results
                      if r["config"]["poison_fraction"] == pf
                      and r["config"]["hidden_dim"] == hd]
            aucs_by_pf.append(np.mean([r["detection_auc"] for r in subset]))
        ax.plot([pf * 100 for pf in poison_fracs], aucs_by_pf,
                marker="o", label=f"hidden={hd}")

    ax.set_xlabel("Poison Fraction (%)")
    ax.set_ylabel("Detection AUC")
    ax.set_title("Detection AUC vs Poison Fraction by Model Size")
    ax.legend()
    ax.set_ylim(0.4, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_auc_by_model_size.png")
    plt.savefig(path, dpi=150)
    plt.close()
    figure_paths.append(path)

    # Figure 3: Eigenvalue ratio by poison fraction
    fig, ax = plt.subplots(figsize=(7, 5))
    ratios_by_pf = []
    stds_by_pf = []
    for pf in poison_fracs:
        subset = [r for r in results if r["config"]["poison_fraction"] == pf]
        rats = [r["eigenvalue_ratio"] for r in subset]
        ratios_by_pf.append(np.mean(rats))
        stds_by_pf.append(np.std(rats))

    x_pos = range(len(poison_fracs))
    ax.bar(x_pos, ratios_by_pf, yerr=stds_by_pf, capsize=5, color="steelblue", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{pf*100:.0f}%" for pf in poison_fracs])
    ax.set_xlabel("Poison Fraction")
    ax.set_ylabel("Eigenvalue Ratio (Top / Second)")
    ax.set_title("Spectral Gap vs Poison Fraction")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_eigenvalue_ratio.png")
    plt.savefig(path, dpi=150)
    plt.close()
    figure_paths.append(path)

    return figure_paths
