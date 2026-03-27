"""Generate publication-quality plots for the poisoning sensitivity study."""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.analysis import AggregatedPoint, SigmoidFit, _sigmoid


COLORS = {32: "#1f77b4", 64: "#ff7f0e", 128: "#2ca02c"}
MARKERS = {32: "o", 64: "s", 128: "^"}


def plot_accuracy_vs_poison(
    agg_points: list[AggregatedPoint],
    fits: list[SigmoidFit],
    output_dir: str,
) -> str:
    """Plot test accuracy vs. poison fraction with sigmoid fits.

    Args:
        agg_points: Aggregated experiment data.
        fits: Sigmoid fits per model size.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    fit_dict = {f.hidden_width: f for f in fits}

    for hw in [32, 64, 128]:
        pts = [p for p in agg_points if p.hidden_width == hw]
        pts.sort(key=lambda p: p.poison_fraction)

        x = [p.poison_fraction for p in pts]
        y = [p.test_acc_mean for p in pts]
        yerr = [p.test_acc_std for p in pts]

        ax.errorbar(
            x, y, yerr=yerr,
            marker=MARKERS[hw], color=COLORS[hw],
            label=f"Width {hw}", capsize=3, linewidth=1.5, markersize=6,
        )

        # Overlay sigmoid fit
        if hw in fit_dict:
            f = fit_dict[hw]
            x_smooth = np.linspace(0, 0.55, 200)
            y_smooth = _sigmoid(x_smooth, f.L, f.k, f.x0, f.b)
            ax.plot(x_smooth, y_smooth, "--", color=COLORS[hw], alpha=0.6,
                    label=f"Fit (k={f.k:.1f}, R²={f.r_squared:.3f})")

            # Mark critical threshold
            if f.threshold_midpoint < 0.6:
                ax.axvline(f.threshold_midpoint, color=COLORS[hw],
                           linestyle=":", alpha=0.4)

    ax.set_xlabel("Poison Fraction", fontsize=12)
    ax.set_ylabel("Clean Test Accuracy", fontsize=12)
    ax.set_title("Data Poisoning Sensitivity: Accuracy vs. Poison Fraction", fontsize=13)
    ax.legend(fontsize=9, loc="lower left")
    ax.set_xlim(-0.02, 0.55)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "accuracy_vs_poison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_generalization_gap(
    agg_points: list[AggregatedPoint],
    output_dir: str,
) -> str:
    """Plot generalization gap vs. poison fraction.

    Args:
        agg_points: Aggregated experiment data.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    for hw in [32, 64, 128]:
        pts = [p for p in agg_points if p.hidden_width == hw]
        pts.sort(key=lambda p: p.poison_fraction)

        x = [p.poison_fraction for p in pts]
        y = [p.gen_gap_mean for p in pts]
        yerr = [p.gen_gap_std for p in pts]

        ax.errorbar(
            x, y, yerr=yerr,
            marker=MARKERS[hw], color=COLORS[hw],
            label=f"Width {hw}", capsize=3, linewidth=1.5, markersize=6,
        )

    ax.set_xlabel("Poison Fraction", fontsize=12)
    ax.set_ylabel("Generalization Gap (Train Acc - Test Acc)", fontsize=12)
    ax.set_title("Generalization Gap vs. Poison Fraction", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.02, 0.55)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "generalization_gap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_train_vs_test(
    agg_points: list[AggregatedPoint],
    output_dir: str,
) -> str:
    """Plot training accuracy (on poisoned labels) vs. test accuracy.

    Args:
        agg_points: Aggregated experiment data.
        output_dir: Directory to save the plot.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)

    for ax, hw in zip(axes, [32, 64, 128]):
        pts = [p for p in agg_points if p.hidden_width == hw]
        pts.sort(key=lambda p: p.poison_fraction)

        x = [p.poison_fraction for p in pts]
        train_y = [p.train_acc_mean for p in pts]
        test_y = [p.test_acc_mean for p in pts]
        clean_y = [p.train_clean_acc_mean for p in pts]

        ax.plot(x, train_y, "o-", color="#d62728", label="Train (poisoned labels)", markersize=5)
        ax.plot(x, test_y, "s-", color="#1f77b4", label="Test (clean)", markersize=5)
        ax.plot(x, clean_y, "^--", color="#9467bd", label="Train (clean labels)", markersize=5)

        ax.set_xlabel("Poison Fraction", fontsize=11)
        ax.set_title(f"Width {hw}", fontsize=12)
        ax.set_xlim(-0.02, 0.55)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Accuracy", fontsize=11)
    fig.suptitle("Training vs. Test Accuracy Under Poisoning", fontsize=13, y=1.02)
    fig.tight_layout()

    path = os.path.join(output_dir, "train_vs_test.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
