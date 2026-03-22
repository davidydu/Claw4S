"""Plotting functions for double descent experiments.

Generates publication-quality plots of model-wise and epoch-wise double
descent curves, noise comparisons, and variance bands.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt


def plot_model_wise(
    results_by_noise: dict[str, list[dict]],
    interpolation_threshold: int,
    output_path: str,
) -> None:
    """Plot model-wise double descent: test loss vs. feature count.

    Args:
        results_by_noise: Dict mapping noise label -> list of sweep dicts.
        interpolation_threshold: Feature count at interpolation threshold.
        output_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#FF9800", "#F44336"]
    noise_labels = sorted(results_by_noise.keys())

    for idx, label in enumerate(noise_labels):
        data = results_by_noise[label]
        widths = [r["width"] for r in data]
        test_losses = [r["test_loss"] for r in data]
        train_losses = [r["train_loss"] for r in data]
        noise_val = label.replace("noise_", "sigma=")
        color = colors[idx % len(colors)]

        ax1.plot(widths, test_losses, "o-", color=color,
                 label=f"Test ({noise_val})", markersize=4)

        ax2.plot(widths, train_losses, "s--", color=color,
                 label=f"Train ({noise_val})", markersize=3, alpha=0.7)
        ax2.plot(widths, test_losses, "o-", color=color,
                 label=f"Test ({noise_val})", markersize=4)

    for ax in [ax1, ax2]:
        ax.axvline(x=interpolation_threshold, color="gray", linestyle=":",
                   alpha=0.7, label=f"Threshold (p={interpolation_threshold})")
        ax.set_xlabel("Number of Random Features (p)", fontsize=12)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax1.set_ylabel("Test MSE (log scale)", fontsize=12)
    ax1.set_title("Model-Wise Double Descent\n(Random Features)", fontsize=13)

    ax2.set_ylabel("MSE (log scale)", fontsize=12)
    ax2.set_title("Train vs. Test Loss", fontsize=13)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_noise_comparison(
    results_by_noise: dict[str, list[dict]],
    interpolation_threshold: int,
    output_path: str,
) -> None:
    """Plot noise comparison: overlaid double descent curves.

    Args:
        results_by_noise: Dict mapping noise label -> list of sweep dicts.
        interpolation_threshold: Feature count at interpolation threshold.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["#2196F3", "#FF9800", "#F44336"]
    noise_labels = sorted(results_by_noise.keys())

    for idx, label in enumerate(noise_labels):
        data = results_by_noise[label]
        widths = [r["width"] for r in data]
        test_losses = [r["test_loss"] for r in data]
        noise_val = label.replace("noise_", "sigma=")
        color = colors[idx % len(colors)]

        ax.plot(widths, test_losses, "o-", color=color,
                label=f"{noise_val}", markersize=5, linewidth=2)

    ax.axvline(x=interpolation_threshold, color="gray", linestyle=":",
               alpha=0.7, linewidth=2,
               label=f"Interp. threshold (p={interpolation_threshold})")

    ax.set_xlabel("Number of Random Features (p)", fontsize=13)
    ax.set_ylabel("Test MSE (log scale)", fontsize=13)
    ax.set_title("Label Noise Amplifies Double Descent", fontsize=14)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_epoch_wise(
    results_by_noise: dict[str, dict],
    output_path: str,
) -> None:
    """Plot epoch-wise double descent: test loss vs. training epoch.

    Args:
        results_by_noise: Dict mapping noise label -> epoch sweep dict.
        output_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#2196F3", "#FF9800", "#F44336"]
    noise_labels = sorted(results_by_noise.keys())
    width = None

    for idx, label in enumerate(noise_labels):
        data = results_by_noise[label]
        epochs = data["epochs"]
        test_losses = data["test_losses"]
        train_losses = data["train_losses"]
        noise_val = label.replace("noise_", "sigma=")
        width = data["width"]
        color = colors[idx % len(colors)]

        ax1.plot(epochs, test_losses, "-", color=color,
                 label=f"Test ({noise_val})", linewidth=1.5)

        ax2.plot(epochs, train_losses, "--", color=color,
                 label=f"Train ({noise_val})", linewidth=1, alpha=0.7)
        ax2.plot(epochs, test_losses, "-", color=color,
                 label=f"Test ({noise_val})", linewidth=1.5)

    for ax in [ax1, ax2]:
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax1.set_ylabel("Test MSE (log scale)", fontsize=12)
    ax1.set_title(f"Epoch-Wise Double Descent (MLP, h={width})", fontsize=13)

    ax2.set_ylabel("MSE (log scale)", fontsize=12)
    ax2.set_title("Train vs. Test Loss Over Epochs", fontsize=13)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mlp_comparison(
    mlp_results: list[dict],
    rf_results: list[dict],
    n_train: int,
    output_path: str,
) -> None:
    """Plot MLP vs random features double descent comparison.

    Args:
        mlp_results: MLP sweep results.
        rf_results: Random features sweep results (highest noise).
        n_train: Number of training samples.
        output_path: Path to save the figure.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Random features (left)
    rf_widths = [r["width"] for r in rf_results]
    rf_test = [r["test_loss"] for r in rf_results]
    rf_train = [r["train_loss"] for r in rf_results]

    ax1.plot(rf_widths, rf_test, "o-", color="#F44336", label="Test", markersize=4)
    ax1.plot(rf_widths, rf_train, "s--", color="#2196F3", label="Train",
             markersize=3, alpha=0.7)
    ax1.axvline(x=n_train, color="gray", linestyle=":", alpha=0.7,
                label=f"p=n={n_train}")
    ax1.set_xlabel("Random Features (p)", fontsize=12)
    ax1.set_ylabel("MSE (log scale)", fontsize=12)
    ax1.set_title("Random Features Model", fontsize=13)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)

    # MLP (right)
    mlp_widths = [r["width"] for r in mlp_results]
    mlp_test = [r["test_loss"] for r in mlp_results]
    mlp_train = [r["train_loss"] for r in mlp_results]
    mlp_ratios = [r["param_ratio"] for r in mlp_results]

    ax2.plot(mlp_widths, mlp_test, "o-", color="#F44336", label="Test", markersize=4)
    ax2.plot(mlp_widths, mlp_train, "s--", color="#2196F3", label="Train",
             markersize=3, alpha=0.7)

    # Mark interpolation threshold for MLP
    mlp_threshold_h = max(1, round((n_train - 1) / 22))  # d=20 -> d+2=22
    ax2.axvline(x=mlp_threshold_h, color="gray", linestyle=":", alpha=0.7,
                label=f"Threshold h~{mlp_threshold_h}")

    ax2.set_xlabel("Hidden Width (h)", fontsize=12)
    ax2.set_ylabel("MSE (log scale)", fontsize=12)
    ax2.set_title("Trained MLP (Adam, 4000 epochs)", fontsize=13)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variance_bands(
    variance_stats: dict,
    interpolation_threshold: int,
    output_path: str,
) -> None:
    """Plot double descent curve with variance bands across seeds.

    Args:
        variance_stats: Dict with widths, mean_test_loss, std_test_loss.
        interpolation_threshold: Feature count at threshold.
        output_path: Path to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    widths = np.array(variance_stats["widths"])
    mean = np.array(variance_stats["mean_test_loss"])
    std = np.array(variance_stats["std_test_loss"])

    ax.plot(widths, mean, "o-", color="#F44336", markersize=4, linewidth=2,
            label="Mean test MSE")
    ax.fill_between(widths, np.maximum(mean - std, 1e-3), mean + std,
                    alpha=0.2, color="#F44336", label="+/- 1 std")

    ax.axvline(x=interpolation_threshold, color="gray", linestyle=":",
               alpha=0.7, linewidth=2,
               label=f"Threshold (p={interpolation_threshold})")

    ax.set_xlabel("Number of Random Features (p)", fontsize=13)
    ax.set_ylabel("Test MSE (log scale)", fontsize=13)
    ax.set_title(
        f"Double Descent with Variance Bands "
        f"({variance_stats['n_seeds']} seeds)",
        fontsize=14,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
