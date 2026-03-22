"""Plotting functions for gradient norm phase transition experiments.

Generates:
  1. Per-run overlay: gradient norms + test metric vs epoch
  2. Summary grid: all runs side by side
  3. Lag bar chart: gradient transition lead time per configuration
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


def plot_single_run(
    result: dict,
    analysis: dict,
    output_dir: str,
) -> str:
    """Plot gradient norms and test metric for a single training run.

    Dual y-axis: left = gradient L2 norm, right = test metric.
    Vertical lines mark detected transitions.

    Returns path to saved figure.
    """
    epochs = result["epochs"]
    task = result["task_name"]
    frac = result["frac"]
    metric_name = result["metric_name"]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot per-layer gradient norms
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (layer_name, norms) in enumerate(result["grad_norms"].items()):
        color = colors[i % len(colors)]
        ax1.plot(epochs, norms, color=color, alpha=0.6, linewidth=1,
                 label=f"grad norm ({layer_name})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Gradient L2 Norm", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")

    # Plot test metric on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(epochs, result["test_metric"], color="#d62728", linewidth=2,
             label=f"test {metric_name}")
    ax2.plot(epochs, result["train_metric"], color="#d62728", linewidth=1,
             linestyle="--", alpha=0.5, label=f"train {metric_name}")
    ax2.set_ylabel(f"Test {metric_name.replace('_', ' ').title()}", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    # Mark transitions
    gnorm_epoch = analysis["gnorm_transition_epoch"]
    metric_epoch = analysis["metric_transition_epoch"]
    ax1.axvline(gnorm_epoch, color="#1f77b4", linestyle=":", linewidth=1.5,
                label=f"grad transition (epoch {gnorm_epoch})")
    ax1.axvline(metric_epoch, color="#d62728", linestyle=":", linewidth=1.5,
                label=f"metric transition (epoch {metric_epoch})")

    lag = analysis["lag_epochs"]
    title = (f"{task} (frac={frac:.0%}) -- "
             f"Gradient leads by {lag} epochs")
    ax1.set_title(title)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)

    plt.tight_layout()
    fname = f"run_{task}_frac{frac:.2f}.png"
    path = os.path.join(output_dir, fname)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_summary_grid(
    results: list[dict],
    analyses: list[dict],
    output_dir: str,
) -> str:
    """Plot all runs in a grid: rows=tasks, cols=fractions.

    Returns path to saved figure.
    """
    tasks = sorted(set(a["task_name"] for a in analyses))
    fracs = sorted(set(a["frac"] for a in analyses))

    n_rows = len(tasks)
    n_cols = len(fracs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)

    for row, task in enumerate(tasks):
        for col, frac in enumerate(fracs):
            ax = axes[row][col]
            # Find matching run
            match = None
            match_r = None
            for r, a in zip(results, analyses):
                if a["task_name"] == task and abs(a["frac"] - frac) < 0.01:
                    match = a
                    match_r = r
                    break

            if match is None or match_r is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            epochs = match_r["epochs"]
            combined_gnorm = match["combined_gnorm"]

            # Normalize for dual plot
            gnorm_arr = np.array(combined_gnorm)
            metric_arr = np.array(match_r["test_metric"])

            gnorm_norm = (gnorm_arr - gnorm_arr.min()) / (gnorm_arr.max() - gnorm_arr.min() + 1e-12)
            metric_norm = (metric_arr - metric_arr.min()) / (metric_arr.max() - metric_arr.min() + 1e-12)

            ax.plot(epochs, gnorm_norm, color="#1f77b4", linewidth=1.5,
                    label="grad norm (normed)")
            ax.plot(epochs, metric_norm, color="#d62728", linewidth=1.5,
                    label=f"test {match_r['metric_name']} (normed)")

            ax.axvline(match["gnorm_transition_epoch"], color="#1f77b4",
                       linestyle=":", linewidth=1)
            ax.axvline(match["metric_transition_epoch"], color="#d62728",
                       linestyle=":", linewidth=1)

            lag = match["lag_epochs"]
            ax.set_title(f"{task} frac={frac:.0%} (lag={lag})", fontsize=10)
            if row == 0 and col == 0:
                ax.legend(fontsize=7, loc="center right")
            if row == n_rows - 1:
                ax.set_xlabel("Epoch")

    plt.tight_layout()
    path = os.path.join(output_dir, "summary_grid.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_lag_barchart(
    analyses: list[dict],
    output_dir: str,
) -> str:
    """Bar chart showing gradient-to-metric lag for each configuration.

    Returns path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    labels = []
    lags = []
    colors = []

    for a in sorted(analyses, key=lambda x: (x["task_name"], x["frac"])):
        label = f"{a['task_name']}\n{a['frac']:.0%}"
        labels.append(label)
        lags.append(a["lag_epochs"])
        colors.append("#2ca02c" if a["lag_epochs"] > 0 else "#d62728")

    bars = ax.bar(range(len(labels)), lags, color=colors, edgecolor="black",
                  linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Lag (epochs): gradient transition before metric transition")
    ax.set_title("Gradient Norm Phase Transition Lead Time")
    ax.axhline(0, color="black", linewidth=0.8)

    # Annotate bars
    for bar, lag in zip(bars, lags):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{lag}", ha="center", va="bottom" if lag >= 0 else "top",
                fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "lag_barchart.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weight_norms(
    results: list[dict],
    analyses: list[dict],
    output_dir: str,
) -> str:
    """Plot weight norm trajectories for all runs.

    Returns path to saved figure.
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for i, (r, a) in enumerate(zip(results, analyses)):
        ax = axes[0][i]
        epochs = r["epochs"]
        for layer_name, norms in r["weight_norms"].items():
            ax.plot(epochs, norms, linewidth=1.5, label=f"{layer_name}")
        ax.axvline(a["gnorm_transition_epoch"], color="gray",
                   linestyle=":", linewidth=1, label="grad transition")
        ax.set_title(f"{a['task_name']} frac={a['frac']:.0%}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Weight L2 Norm")
        ax.legend(fontsize=7)

    plt.tight_layout()
    path = os.path.join(output_dir, "weight_norms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
