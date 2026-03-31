"""Plotting functions for symmetry-breaking experiment results."""

import os
from typing import Dict, List, Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import numpy as np


def plot_symmetry_trajectories(
    results: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Plot symmetry metric vs epoch for all runs, grouped by hidden dim.

    Creates one subplot per hidden dimension, with lines colored by epsilon.

    Args:
        results: List of per-run result dicts from trainer.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    hidden_dims = sorted(set(r["hidden_dim"] for r in results))
    epsilons = sorted(set(r["epsilon"] for r in results))

    fig, axes = plt.subplots(1, len(hidden_dims), figsize=(5 * len(hidden_dims), 4))
    if len(hidden_dims) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(epsilons)))

    for ax, hd in zip(axes, hidden_dims):
        for eps, color in zip(epsilons, colors):
            matching = [
                r for r in results
                if r["hidden_dim"] == hd and r["epsilon"] == eps
            ]
            if not matching:
                continue
            r = matching[0]
            label = f"eps={eps:.0e}" if eps > 0 else "eps=0"
            ax.plot(
                r["epochs_logged"],
                r["symmetry_values"],
                label=label,
                color=color,
                linewidth=1.5,
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Symmetry Metric (mean cos sim)")
        ax.set_title(f"Hidden dim = {hd}")
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Symmetry Breaking During Training", fontsize=14)
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "symmetry_trajectories.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved symmetry trajectories plot to {path}")
    return path


def plot_accuracy_vs_epsilon(
    results: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Plot final test accuracy vs epsilon, grouped by hidden dim.

    Args:
        results: List of per-run result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    hidden_dims = sorted(set(r["hidden_dim"] for r in results))

    fig, ax = plt.subplots(figsize=(7, 5))

    for hd in hidden_dims:
        matching = sorted(
            [r for r in results if r["hidden_dim"] == hd],
            key=lambda r: r["epsilon"],
        )
        epsilons = [r["epsilon"] for r in matching]
        accs = [r["final_test_acc"] for r in matching]

        # Replace 0 with a small value for log scale
        plot_eps = [max(e, 1e-8) for e in epsilons]
        ax.plot(plot_eps, accs, "o-", label=f"hidden={hd}", linewidth=1.5)

    ax.set_xscale("log")
    ax.set_xlabel("Epsilon (init perturbation scale)")
    ax.set_ylabel("Final Test Accuracy")
    ax.set_title("Test Accuracy vs Initialization Perturbation")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "accuracy_vs_epsilon.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved accuracy vs epsilon plot to {path}")
    return path


def plot_final_symmetry_heatmap(
    results: List[Dict[str, Any]],
    output_dir: str = "results",
) -> str:
    """Plot heatmap of final symmetry values (hidden_dim x epsilon).

    Args:
        results: List of per-run result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved figure.
    """
    hidden_dims = sorted(set(r["hidden_dim"] for r in results))
    epsilons = sorted(set(r["epsilon"] for r in results))

    data = np.full((len(hidden_dims), len(epsilons)), np.nan)

    for r in results:
        i = hidden_dims.index(r["hidden_dim"])
        j = epsilons.index(r["epsilon"])
        data[i, j] = r["final_symmetry"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(epsilons)))
    ax.set_xticklabels([f"{e:.0e}" if e > 0 else "0" for e in epsilons])
    ax.set_yticks(range(len(hidden_dims)))
    ax.set_yticklabels([str(hd) for hd in hidden_dims])
    ax.set_xlabel("Epsilon")
    ax.set_ylabel("Hidden Dimension")
    ax.set_title("Final Symmetry Metric (lower = more broken)")

    # Add text annotations
    for i in range(len(hidden_dims)):
        for j in range(len(epsilons)):
            val = data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)

    fig.colorbar(im, label="Symmetry Metric")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "symmetry_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved symmetry heatmap to {path}")
    return path
