"""Visualization of eigenvalue spectra and RMT analysis results."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.rmt_analysis import marchenko_pastur_pdf


def plot_eigenvalue_spectra(
    all_results: list[dict],
    save_path: str = "results/eigenvalue_spectra.png",
) -> str:
    """Plot eigenvalue histograms with MP overlay for all analyzed layers.

    Creates a grid of subplots, one per (model, layer) combination.

    Args:
        all_results: List of analysis dicts from analyze_weight_matrix.
        save_path: Path to save the figure.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Group results by model_label
    models = {}
    for r in all_results:
        label = r.get("model_label", "unknown")
        if label not in models:
            models[label] = []
        models[label].append(r)

    n_models = len(models)
    # Find max layers across models
    max_layers = max(len(layers) for layers in models.values())

    fig, axes = plt.subplots(
        n_models, max_layers,
        figsize=(5 * max_layers, 4 * n_models),
        squeeze=False,
    )

    for row, (model_label, layers) in enumerate(sorted(models.items())):
        for col, layer_result in enumerate(layers):
            ax = axes[row, col]
            eigenvalues = np.array(layer_result["eigenvalues"])
            gamma = layer_result["gamma"]
            sigma_sq = layer_result["sigma_sq"]
            lam_min = layer_result["lambda_minus"]
            lam_max = layer_result["lambda_plus"]
            ks = layer_result["ks_statistic"]

            # Histogram of empirical eigenvalues
            n_bins = min(50, max(10, len(eigenvalues) // 3))
            ax.hist(
                eigenvalues, bins=n_bins, density=True,
                alpha=0.6, color="steelblue", edgecolor="white",
                label="Empirical",
            )

            # MP theoretical PDF overlay
            x_range = np.linspace(
                max(0, lam_min - 0.1 * (lam_max - lam_min)),
                lam_max + 0.3 * (lam_max - lam_min),
                500,
            )
            mp_pdf = marchenko_pastur_pdf(x_range, gamma, sigma_sq)
            ax.plot(x_range, mp_pdf, "r-", linewidth=2, label="MP theory")

            # Mark bulk edges
            ax.axvline(lam_min, color="green", linestyle="--", alpha=0.7,
                       label=f"$\\lambda_-$={lam_min:.3f}")
            ax.axvline(lam_max, color="orange", linestyle="--", alpha=0.7,
                       label=f"$\\lambda_+$={lam_max:.3f}")

            ax.set_title(
                f"{model_label} / {layer_result['layer_name']}\n"
                f"KS={ks:.3f}, outliers={layer_result['outlier_fraction']:.2f}",
                fontsize=9,
            )
            ax.set_xlabel("Eigenvalue")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7, loc="upper right")

        # Hide unused axes
        for col in range(len(layers), max_layers):
            axes[row, col].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_ks_summary(
    all_results: list[dict],
    save_path: str = "results/ks_summary.png",
) -> str:
    """Plot KS statistics across layers and model configurations.

    Args:
        all_results: List of analysis dicts.
        save_path: Path to save the figure.

    Returns:
        Path to the saved figure.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Group by model_label
    models = {}
    for r in all_results:
        label = r.get("model_label", "unknown")
        if label not in models:
            models[label] = []
        models[label].append(r)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Sort model labels for consistent ordering
    sorted_labels = sorted(models.keys())
    x = np.arange(len(sorted_labels))
    width = 0.25

    # Collect layer names from first model
    first_layers = models[sorted_labels[0]]
    layer_names = [r["layer_name"] for r in first_layers]

    # Plot 1: KS statistics by layer
    ax = axes[0]
    for i, layer_name in enumerate(layer_names):
        ks_vals = []
        for label in sorted_labels:
            layer_results = [r for r in models[label] if r["layer_name"] == layer_name]
            ks_vals.append(layer_results[0]["ks_statistic"] if layer_results else 0)
        ax.bar(x + i * width, ks_vals, width, label=layer_name, alpha=0.8)
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("KS Statistic")
    ax.set_title("KS Distance from Marchenko-Pastur")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)

    # Plot 2: Outlier fractions
    ax = axes[1]
    for i, layer_name in enumerate(layer_names):
        outlier_vals = []
        for label in sorted_labels:
            layer_results = [r for r in models[label] if r["layer_name"] == layer_name]
            outlier_vals.append(
                layer_results[0]["outlier_fraction"] if layer_results else 0
            )
        ax.bar(x + i * width, outlier_vals, width, label=layer_name, alpha=0.8)
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Outlier Fraction")
    ax.set_title("Eigenvalues Outside MP Bulk")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)

    # Plot 3: Spectral norm ratio
    ax = axes[2]
    for i, layer_name in enumerate(layer_names):
        snr_vals = []
        for label in sorted_labels:
            layer_results = [r for r in models[label] if r["layer_name"] == layer_name]
            snr_vals.append(
                layer_results[0]["spectral_norm_ratio"] if layer_results else 0
            )
        ax.bar(x + i * width, snr_vals, width, label=layer_name, alpha=0.8)
    ax.axhline(1.0, color="red", linestyle="--", alpha=0.5, label="MP edge")
    ax.set_xlabel("Model Configuration")
    ax.set_ylabel("Spectral Norm Ratio")
    ax.set_title("Max Eigenvalue / MP Upper Bound")
    ax.set_xticks(x + width)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right", fontsize=7)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
