"""Visualization module for DP noise calibration comparison.

Generates three types of figures:
1. Heatmaps of tightness ratios for each method
2. Epsilon vs T curves at fixed sigma
3. Method comparison bar charts
"""

import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Consistent color scheme for methods
METHOD_COLORS = {
    "naive": "#e74c3c",      # red
    "advanced": "#f39c12",   # orange
    "rdp": "#2ecc71",        # green
    "gdp": "#3498db",        # blue
}

METHOD_LABELS = {
    "naive": "Naive Composition",
    "advanced": "Advanced Composition",
    "rdp": "Renyi DP (RDP)",
    "gdp": "Gaussian DP (GDP)",
}


def generate_all_figures(data: dict, output_dir: str = "results") -> list[str]:
    """Generate all visualization figures and return list of saved paths."""
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    paths.append(_plot_epsilon_vs_T(data, output_dir))
    paths.append(_plot_tightness_heatmap(data, output_dir))
    paths.append(_plot_method_comparison_bars(data, output_dir))
    paths.append(_plot_epsilon_vs_sigma(data, output_dir))

    return paths


def _plot_epsilon_vs_T(data: dict, output_dir: str) -> str:
    """Plot epsilon vs T for each method at fixed sigma and delta."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Privacy Loss (epsilon) vs Composition Steps (T)",
                 fontsize=14, fontweight="bold")

    T_values = data["grid"]["T_values"]
    sigma_values = data["grid"]["sigma_values"]
    delta = 1e-6  # fixed delta for this plot

    for idx, sigma in enumerate(sigma_values):
        ax = axes[idx // 3][idx % 3]
        for method in data["grid"]["methods"]:
            eps_values = []
            for T in T_values:
                for r in data["results"]:
                    if r["T"] == T and r["sigma"] == sigma and r["delta"] == delta:
                        eps = r["epsilons"][method]
                        if eps == "Infinity" or (isinstance(eps, float) and eps == float("inf")):
                            eps = None
                        eps_values.append(eps)
                        break

            # Filter out None values for plotting
            valid_T = [t for t, e in zip(T_values, eps_values) if e is not None]
            valid_eps = [e for e in eps_values if e is not None]

            if valid_eps:
                ax.plot(valid_T, valid_eps,
                        marker="o", linewidth=2, markersize=5,
                        color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Composition Steps (T)")
        ax.set_ylabel("Privacy Loss (epsilon)")
        ax.set_title(f"sigma = {sigma}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "epsilon_vs_T.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_tightness_heatmap(data: dict, output_dir: str) -> str:
    """Plot heatmap of tightness ratios: method_eps / best_eps."""
    methods = data["grid"]["methods"]
    T_values = data["grid"]["T_values"]
    sigma_values = data["grid"]["sigma_values"]
    delta = 1e-6  # fixed delta

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Tightness Ratio (method_eps / best_eps) at delta=1e-6",
                 fontsize=14, fontweight="bold")

    for m_idx, method in enumerate(methods):
        ax = axes[m_idx]
        matrix = np.full((len(T_values), len(sigma_values)), np.nan)

        for i, T in enumerate(T_values):
            for j, sigma in enumerate(sigma_values):
                for r in data["results"]:
                    if r["T"] == T and r["sigma"] == sigma and r["delta"] == delta:
                        ratio = r["tightness_ratio"].get(method, None)
                        if ratio is not None and ratio != "Infinity":
                            matrix[i, j] = float(ratio)
                        break

        im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r",
                        vmin=1.0, vmax=10.0)
        ax.set_xticks(range(len(sigma_values)))
        ax.set_xticklabels([str(s) for s in sigma_values], fontsize=8)
        ax.set_yticks(range(len(T_values)))
        ax.set_yticklabels([str(t) for t in T_values], fontsize=8)
        ax.set_xlabel("Noise Multiplier (sigma)")
        ax.set_ylabel("Composition Steps (T)")
        ax.set_title(METHOD_LABELS[method], fontsize=10)

        # Annotate cells
        for i in range(len(T_values)):
            for j in range(len(sigma_values)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text = f"{val:.1f}x" if val < 100 else ">100x"
                    color = "white" if val > 5 else "black"
                    ax.text(j, i, text, ha="center", va="center",
                            fontsize=7, color=color)

    fig.colorbar(im, ax=axes, label="Tightness Ratio", shrink=0.8)
    plt.tight_layout()
    path = os.path.join(output_dir, "tightness_heatmap.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_method_comparison_bars(data: dict, output_dir: str) -> str:
    """Bar chart showing which method wins most often, by T value."""
    summary = data["summary"]
    methods = data["grid"]["methods"]
    T_values = data["grid"]["T_values"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall win counts
    ax = axes[0]
    wins = [summary["win_counts"].get(m, 0) for m in methods]
    colors = [METHOD_COLORS[m] for m in methods]
    labels = [METHOD_LABELS[m] for m in methods]
    bars = ax.bar(labels, wins, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Number of Wins (tightest bound)")
    ax.set_title("Overall Method Win Counts")
    for bar, w in zip(bars, wins):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.3,
                str(w), ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0, max(wins) * 1.15 if max(wins) > 0 else 1)
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    # Right: average tightness ratio
    ax = axes[1]
    avg_tight = [summary["avg_tightness_ratio"].get(m, 0) for m in methods]
    std_tight = [summary["std_tightness_ratio"].get(m, 0) for m in methods]
    bars = ax.bar(labels, avg_tight, yerr=std_tight,
                  color=colors, edgecolor="black", linewidth=0.5,
                  capsize=4)
    ax.set_ylabel("Average Tightness Ratio")
    ax.set_title("Mean Tightness Ratio (lower = tighter)")
    ax.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="Optimal (1.0)")
    for bar, a in zip(bars, avg_tight):
        val = a if a != float("inf") else 999
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                f"{val:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.legend()
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

    plt.tight_layout()
    path = os.path.join(output_dir, "method_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_epsilon_vs_sigma(data: dict, output_dir: str) -> str:
    """Plot epsilon vs sigma for each method at fixed T and delta."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Privacy Loss (epsilon) vs Noise Multiplier (sigma) at delta=1e-6",
                 fontsize=14, fontweight="bold")

    sigma_values = data["grid"]["sigma_values"]
    T_values = data["grid"]["T_values"]
    delta = 1e-6

    for idx, T in enumerate(T_values):
        ax = axes[idx // 2][idx % 2]
        for method in data["grid"]["methods"]:
            eps_values = []
            for sigma in sigma_values:
                for r in data["results"]:
                    if r["T"] == T and r["sigma"] == sigma and r["delta"] == delta:
                        eps = r["epsilons"][method]
                        if eps == "Infinity" or (isinstance(eps, float) and eps == float("inf")):
                            eps = None
                        eps_values.append(eps)
                        break

            valid_sigma = [s for s, e in zip(sigma_values, eps_values) if e is not None]
            valid_eps = [e for e in eps_values if e is not None]

            if valid_eps:
                ax.plot(valid_sigma, valid_eps,
                        marker="s", linewidth=2, markersize=5,
                        color=METHOD_COLORS[method],
                        label=METHOD_LABELS[method])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Noise Multiplier (sigma)")
        ax.set_ylabel("Privacy Loss (epsilon)")
        ax.set_title(f"T = {T}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "epsilon_vs_sigma.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
