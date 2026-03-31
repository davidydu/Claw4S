"""Visualization: phase diagrams and transfer heatmaps."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: Path) -> dict:
    """Load transfer_results.json from results directory.

    Args:
        results_dir: Path to results directory.

    Returns:
        Parsed JSON dict.

    Raises:
        FileNotFoundError: If transfer_results.json is missing.
    """
    path = results_dir / "transfer_results.json"
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    with open(path) as f:
        return json.load(f)


def plot_transfer_heatmap(results: dict, results_dir: Path) -> Path:
    """Plot 4x4 heatmap of mean transfer rates (same architecture).

    Args:
        results: Full results dict from transfer_results.json.
        results_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    widths = results["config"]["widths"]
    n = len(widths)

    # Build mean transfer rate matrix
    grid = np.zeros((n, n))
    counts = np.zeros((n, n))
    for r in results["same_arch_results"]:
        i = widths.index(r["source_width"])
        j = widths.index(r["target_width"])
        grid[i, j] += r["transfer_rate"]
        counts[i, j] += 1

    # Avoid division by zero
    counts = np.where(counts == 0, 1, counts)
    grid = grid / counts

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")

    # Annotate cells
    for i in range(n):
        for j in range(n):
            color = "white" if grid[i, j] > 0.6 else "black"
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(w) for w in widths])
    ax.set_xlabel("Target Width", fontsize=12)
    ax.set_ylabel("Source Width", fontsize=12)
    ax.set_title("Adversarial Transfer Rate (FGSM, same architecture)", fontsize=13)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Transfer Rate", fontsize=11)

    out_path = results_dir / "transfer_heatmap.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_transfer_by_ratio(results: dict, results_dir: Path) -> Path:
    """Plot transfer rate vs capacity ratio (target_width / source_width).

    Shows individual points and mean trend with error bars.

    Args:
        results: Full results dict.
        results_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    # Collect (ratio, transfer_rate) pairs
    ratios = []
    rates = []
    for r in results["same_arch_results"]:
        ratios.append(r["capacity_ratio"])
        rates.append(r["transfer_rate"])

    ratios = np.array(ratios)
    rates = np.array(rates)

    # Unique ratios for aggregation
    unique_ratios = sorted(set(ratios))
    means = [np.mean(rates[ratios == r]) for r in unique_ratios]
    stds = [np.std(rates[ratios == r]) for r in unique_ratios]

    fig, ax = plt.subplots(figsize=(8, 5))

    # Scatter individual points
    ax.scatter(ratios, rates, alpha=0.3, color="steelblue", s=40, label="Individual runs")

    # Mean + std
    ax.errorbar(unique_ratios, means, yerr=stds, fmt="o-", color="darkred",
                linewidth=2, markersize=8, capsize=4, label="Mean +/- std")

    ax.axvline(x=1.0, color="gray", linestyle="--", alpha=0.5, label="Ratio = 1")
    ax.set_xlabel("Capacity Ratio (Target Width / Source Width)", fontsize=12)
    ax.set_ylabel("Transfer Rate", fontsize=12)
    ax.set_title("Adversarial Transferability vs Capacity Ratio", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.set_xticks(unique_ratios)
    ax.set_xticklabels([f"{r:.2f}" for r in unique_ratios])
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = results_dir / "transfer_by_ratio.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def plot_depth_comparison(results: dict, results_dir: Path) -> Path:
    """Plot same-depth vs cross-depth transfer rates side by side.

    For each (source_width, target_width) pair where source_width == target_width,
    compare 2L->2L transfer to 2L->4L transfer.

    Args:
        results: Full results dict.
        results_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    widths = results["config"]["widths"]

    # Same-depth (diagonal): source_width == target_width, both 2-layer
    same_depth = {}
    for r in results["same_arch_results"]:
        if r["source_width"] == r["target_width"]:
            w = r["source_width"]
            same_depth.setdefault(w, []).append(r["transfer_rate"])

    # Cross-depth (diagonal): source_width == target_width, 2L->4L
    cross_depth = {}
    for r in results["cross_depth_results"]:
        if r["source_width"] == r["target_width"]:
            w = r["source_width"]
            cross_depth.setdefault(w, []).append(r["transfer_rate"])

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(widths))
    bar_width = 0.35

    same_means = [np.mean(same_depth.get(w, [0])) for w in widths]
    same_stds = [np.std(same_depth.get(w, [0])) for w in widths]
    cross_means = [np.mean(cross_depth.get(w, [0])) for w in widths]
    cross_stds = [np.std(cross_depth.get(w, [0])) for w in widths]

    ax.bar(x - bar_width / 2, same_means, bar_width, yerr=same_stds,
           label="Same depth (2L -> 2L)", color="steelblue", capsize=4)
    ax.bar(x + bar_width / 2, cross_means, bar_width, yerr=cross_stds,
           label="Cross depth (2L -> 4L)", color="coral", capsize=4)

    ax.set_xlabel("Width (same for source and target)", fontsize=12)
    ax.set_ylabel("Transfer Rate", fontsize=12)
    ax.set_title("Same-Width Transfer: Same Depth vs Cross Depth", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([str(w) for w in widths])
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")

    out_path = results_dir / "depth_comparison.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def generate_all_plots(results_dir: Path) -> list[Path]:
    """Generate all visualization plots.

    Args:
        results_dir: Directory containing transfer_results.json and where
                     plots will be saved.

    Returns:
        List of paths to generated plot files.
    """
    results = load_results(results_dir)
    paths = [
        plot_transfer_heatmap(results, results_dir),
        plot_transfer_by_ratio(results, results_dir),
        plot_depth_comparison(results, results_dir),
    ]
    print(f"\nGenerated {len(paths)} plots.")
    return paths
