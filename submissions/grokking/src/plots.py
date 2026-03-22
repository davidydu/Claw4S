"""Visualization for grokking phase diagrams.

Generates heatmaps showing learning phases across hyperparameter space,
and example training curves illustrating the grokking phenomenon.
"""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import Phase


# Color mapping for phases
PHASE_COLORS = {
    Phase.CONFUSION.value: 0,
    Phase.MEMORIZATION.value: 1,
    Phase.GROKKING.value: 2,
    Phase.COMPREHENSION.value: 3,
}

PHASE_CMAP_COLORS = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]
PHASE_LABELS = ["Confusion", "Memorization", "Grokking", "Comprehension"]


def plot_phase_diagram(
    results: list[dict],
    hidden_dim: int,
    save_path: str,
    weight_decays: list[float] | None = None,
    dataset_fractions: list[float] | None = None,
) -> None:
    """Plot a 2D phase diagram heatmap for a given hidden dimension.

    X-axis: dataset fraction, Y-axis: weight decay.
    Each cell colored by phase classification.

    Args:
        results: List of sweep result dicts.
        hidden_dim: Which hidden_dim slice to plot.
        save_path: Where to save the figure.
        weight_decays: Weight decay values (for axis ticks).
        dataset_fractions: Dataset fraction values (for axis ticks).
    """
    # Filter results for this hidden_dim
    filtered = [r for r in results if r["config"]["hidden_dim"] == hidden_dim]
    if not filtered:
        print(f"  No results for hidden_dim={hidden_dim}, skipping plot.")
        return

    # Determine grid axes
    if weight_decays is None:
        weight_decays = sorted(set(r["config"]["weight_decay"] for r in filtered))
    if dataset_fractions is None:
        dataset_fractions = sorted(
            set(r["config"]["train_fraction"] for r in filtered)
        )

    n_wd = len(weight_decays)
    n_frac = len(dataset_fractions)

    # Build grid
    grid = np.full((n_wd, n_frac), -1, dtype=int)
    wd_to_idx = {wd: i for i, wd in enumerate(weight_decays)}
    frac_to_idx = {f: i for i, f in enumerate(dataset_fractions)}

    for r in filtered:
        wd = r["config"]["weight_decay"]
        frac = r["config"]["train_fraction"]
        phase_val = r["phase"]
        if wd in wd_to_idx and frac in frac_to_idx:
            grid[wd_to_idx[wd], frac_to_idx[frac]] = PHASE_COLORS.get(
                phase_val, -1
            )

    # Create custom colormap
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(PHASE_CMAP_COLORS)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(
        grid,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=0,
        vmax=3,
        interpolation="nearest",
    )

    # Axis labels
    ax.set_xticks(range(n_frac))
    ax.set_xticklabels([f"{f:.0%}" for f in dataset_fractions])
    ax.set_xlabel("Dataset Fraction (train %)")

    ax.set_yticks(range(n_wd))
    wd_labels = []
    for wd in weight_decays:
        if wd == 0:
            wd_labels.append("0")
        elif wd < 0.01:
            wd_labels.append(f"{wd:.0e}")
        else:
            wd_labels.append(f"{wd}")
    ax.set_yticklabels(wd_labels)
    ax.set_ylabel("Weight Decay")

    ax.set_title(f"Grokking Phase Diagram (hidden_dim={hidden_dim})")

    # Add text annotations
    for i in range(n_wd):
        for j in range(n_frac):
            if grid[i, j] >= 0:
                phase_name = PHASE_LABELS[grid[i, j]]
                # Use white text on dark backgrounds
                text_color = "white" if grid[i, j] in [0, 3] else "black"
                ax.text(
                    j, i, phase_name[0],  # First letter of phase
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color=text_color,
                )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=PHASE_CMAP_COLORS[i], label=PHASE_LABELS[i])
        for i in range(4)
    ]
    ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1))

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved phase diagram: {save_path}")


def plot_grokking_curves(
    results: list[dict],
    save_path: str,
    max_curves: int = 4,
) -> None:
    """Plot example training curves showing grokking phenomenon.

    Selects runs classified as GROKKING and plots their train/test accuracy
    over epochs.

    Args:
        results: List of sweep result dicts.
        save_path: Where to save the figure.
        max_curves: Maximum number of example curves to plot.
    """
    # Find grokking runs
    grokking_runs = [r for r in results if r["phase"] == Phase.GROKKING.value]

    if not grokking_runs:
        # Fall back to any interesting runs
        grokking_runs = [
            r for r in results
            if r["metrics"]["final_train_acc"] > 0.5
        ]

    if not grokking_runs:
        print("  No suitable runs for training curves, skipping.")
        return

    # Sort by grokking gap (largest first) and take top max_curves
    grokking_runs.sort(
        key=lambda r: r.get("grokking_gap") or 0, reverse=True
    )
    selected = grokking_runs[:max_curves]

    n = len(selected)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4), squeeze=False)

    for idx, run in enumerate(selected):
        ax = axes[0, idx]
        epochs = run["metrics"]["logged_epochs"]
        train_accs = run["metrics"]["train_accs"]
        test_accs = run["metrics"]["test_accs"]

        ax.plot(epochs, train_accs, "b-", label="Train Acc", linewidth=1.5)
        ax.plot(epochs, test_accs, "r-", label="Test Acc", linewidth=1.5)
        ax.axhline(y=0.95, color="gray", linestyle="--", alpha=0.5, label="95%")

        # Mark milestone epochs
        if run["metrics"]["epoch_train_95"] is not None:
            ax.axvline(
                x=run["metrics"]["epoch_train_95"],
                color="blue", linestyle=":", alpha=0.5,
            )
        if run["metrics"]["epoch_test_95"] is not None:
            ax.axvline(
                x=run["metrics"]["epoch_test_95"],
                color="red", linestyle=":", alpha=0.5,
            )

        hd = run["config"]["hidden_dim"]
        wd = run["config"]["weight_decay"]
        frac = run["config"]["train_fraction"]
        gap = run.get("grokking_gap", "N/A")

        ax.set_title(f"h={hd}, wd={wd}, f={frac}\ngap={gap}", fontsize=10)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    fig.suptitle("Example Grokking Training Curves", fontsize=14, y=1.02)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved training curves: {save_path}")
