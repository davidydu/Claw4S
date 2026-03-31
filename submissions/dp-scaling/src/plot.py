"""Plotting utilities for scaling law visualization.

Generates publication-quality figures comparing scaling behavior
across privacy levels.
"""

import os
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.scaling import power_law


def plot_scaling_laws(results: dict, output_dir: str = "results") -> str:
    """Plot scaling law curves for all privacy levels.

    Creates a log-log plot of test loss vs parameter count with:
    - Data points (mean +/- std across seeds) for each privacy level
    - Fitted power law curves
    - Legend with scaling exponents

    Args:
        results: Full experiment results dictionary.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    colors = {
        "non_private": "#2196F3",
        "moderate_dp": "#FF9800",
        "strong_dp": "#F44336",
    }
    labels = {
        "non_private": "Non-Private (SGD)",
        "moderate_dp": r"Moderate DP ($\sigma$=1.0)",
        "strong_dp": r"Strong DP ($\sigma$=3.0)",
    }

    for privacy_level in ["non_private", "moderate_dp", "strong_dp"]:
        agg = results["aggregated"][privacy_level]
        fit = results["scaling_fits"][privacy_level]

        param_counts = np.array(agg["param_counts"])
        mean_losses = np.array(agg["mean_losses"])
        std_losses = np.array(agg["std_losses"])

        # Data points with error bars
        ax.errorbar(
            param_counts,
            mean_losses,
            yerr=std_losses,
            fmt="o",
            color=colors[privacy_level],
            markersize=8,
            capsize=4,
            label=None,
        )

        # Fitted curve
        if fit.get("alpha") is not None:
            n_smooth = np.logspace(
                np.log10(param_counts.min() * 0.8),
                np.log10(param_counts.max() * 1.2),
                100,
            )
            l_smooth = power_law(n_smooth, fit["a"], fit["alpha"], fit["l_inf"])
            label_str = (
                f'{labels[privacy_level]}: '
                f'$\\alpha$={fit["alpha"]:.3f} '
                f'($R^2$={fit["r_squared"]:.3f})'
            )
            ax.plot(
                n_smooth,
                l_smooth,
                "-",
                color=colors[privacy_level],
                linewidth=2,
                label=label_str,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Parameters", fontsize=13)
    ax.set_ylabel("Test Loss (Cross-Entropy)", fontsize=13)
    ax.set_title("Scaling Laws Under Differential Privacy", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    path = os.path.join(output_dir, "scaling_laws.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {path}")
    return path


def plot_accuracy_comparison(results: dict, output_dir: str = "results") -> str:
    """Plot accuracy vs model size for all privacy levels.

    Args:
        results: Full experiment results dictionary.
        output_dir: Directory to save the figure.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {
        "non_private": "#2196F3",
        "moderate_dp": "#FF9800",
        "strong_dp": "#F44336",
    }
    labels = {
        "non_private": "Non-Private (SGD)",
        "moderate_dp": r"Moderate DP ($\sigma$=1.0)",
        "strong_dp": r"Strong DP ($\sigma$=3.0)",
    }

    for privacy_level in ["non_private", "moderate_dp", "strong_dp"]:
        agg = results["aggregated"][privacy_level]
        param_counts = np.array(agg["param_counts"])
        mean_accs = np.array(agg["mean_accuracies"])

        ax.plot(
            param_counts,
            mean_accs * 100,
            "o-",
            color=colors[privacy_level],
            label=labels[privacy_level],
            markersize=8,
            linewidth=2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Number of Parameters", fontsize=13)
    ax.set_ylabel("Test Accuracy (%)", fontsize=13)
    ax.set_title("Accuracy vs Model Size Under Differential Privacy", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which="both")
    ax.tick_params(labelsize=11)

    fig.tight_layout()
    path = os.path.join(output_dir, "accuracy_comparison.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {path}")
    return path


def generate_all_plots(results_path: str = "results/experiment_results.json") -> None:
    """Generate all figures from saved results.

    Args:
        results_path: Path to the saved experiment results JSON.
    """
    with open(results_path) as f:
        results = json.load(f)

    output_dir = os.path.dirname(results_path)
    plot_scaling_laws(results, output_dir)
    plot_accuracy_comparison(results, output_dir)
    print("All plots generated.")
