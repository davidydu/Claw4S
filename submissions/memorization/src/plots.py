# src/plots.py
"""Plotting functions for memorization capacity results."""

import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments
import matplotlib.pyplot as plt
import numpy as np

from src.analysis import sigmoid


def plot_memorization_curves(analysis: dict, output_dir: str = "results/figures") -> str:
    """Plot train accuracy vs log(#params) for both label types.

    Args:
        analysis: Output from analysis.analyze_results().
        output_dir: Directory to save figures.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    colors = {"random": "#e74c3c", "structured": "#2ecc71"}
    markers = {"random": "o", "structured": "s"}

    for label_type, lt_data in analysis["label_types"].items():
        params = lt_data["params"]
        train_accs = lt_data["train_accs"]
        log_params = np.log10(params)

        # Plot data points
        ax.plot(
            log_params, train_accs,
            marker=markers[label_type],
            color=colors[label_type],
            label=f"{label_type} labels (data)",
            linewidth=2,
            markersize=8,
            zorder=3,
        )

        # Plot sigmoid fit
        sig = lt_data["sigmoid_fit"]
        if sig["fit_success"]:
            x_fine = np.linspace(log_params.min() - 0.2, log_params.max() + 0.2, 200)
            chance = analysis["chance_level"]
            y_norm = sigmoid(x_fine, sig["threshold_log10"], sig["sharpness"])
            y_fit = y_norm * (1.0 - chance) + chance
            ax.plot(
                x_fine, y_fit,
                color=colors[label_type],
                linestyle="--",
                alpha=0.6,
                label=f"{label_type} labels (sigmoid fit, R²={sig['r_squared']:.3f})",
            )

            # Mark threshold
            ax.axvline(
                sig["threshold_log10"],
                color=colors[label_type],
                linestyle=":",
                alpha=0.4,
            )

    # Reference lines
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3, label="100% accuracy")
    ax.axhline(analysis["chance_level"], color="gray", linestyle=":", alpha=0.3,
               label=f"Chance level ({analysis['chance_level']:.0%})")

    # Mark n_train
    n_train = analysis["n_train"]
    ax.axvline(np.log10(n_train), color="blue", linestyle="--", alpha=0.3,
               label=f"#params = n_train ({n_train})")

    ax.set_xlabel("log₁₀(#parameters)", fontsize=12)
    ax.set_ylabel("Training Accuracy", fontsize=12)
    ax.set_title("Memorization Capacity vs. Model Size", fontsize=14)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "memorization_curve.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_threshold_comparison(analysis: dict, output_dir: str = "results/figures") -> str:
    """Plot comparison of interpolation thresholds and test accuracy.

    Args:
        analysis: Output from analysis.analyze_results().
        output_dir: Directory to save figures.

    Returns:
        Path to saved figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = {"random": "#e74c3c", "structured": "#2ecc71"}
    markers = {"random": "o", "structured": "s"}

    # Left panel: train acc vs params
    for label_type, lt_data in analysis["label_types"].items():
        params = lt_data["params"]
        log_params = np.log10(params)
        ax1.plot(
            log_params, lt_data["train_accs"],
            marker=markers[label_type], color=colors[label_type],
            label=f"{label_type} (train)", linewidth=2, markersize=7,
        )
        ax1.plot(
            log_params, lt_data["test_accs"],
            marker=markers[label_type], color=colors[label_type],
            label=f"{label_type} (test)", linewidth=1, markersize=5,
            linestyle=":", alpha=0.6,
        )

    ax1.axhline(analysis["chance_level"], color="gray", linestyle=":", alpha=0.3)
    ax1.set_xlabel("log₁₀(#parameters)", fontsize=11)
    ax1.set_ylabel("Accuracy", fontsize=11)
    ax1.set_title("Train vs. Test Accuracy", fontsize=12)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Right panel: convergence speed
    for label_type, lt_data in analysis["label_types"].items():
        lt_results = [r for r in analysis["label_types"][label_type].get("_raw_results", [])]
        # Use params and epoch data directly from the analysis
        params = lt_data["params"]
        log_params = np.log10(params)

        # We need epoch data from the sweep results - use train_accs as proxy
        # for visualization: higher acc = fewer epochs needed (conceptually)
        ax2.plot(
            log_params, lt_data["train_accs"],
            marker=markers[label_type], color=colors[label_type],
            label=f"{label_type} labels", linewidth=2, markersize=7,
        )

    # Add sigmoid fit parameters as text
    text_lines = []
    for label_type in ["random", "structured"]:
        lt = analysis["label_types"][label_type]
        sig = lt["sigmoid_fit"]
        if sig["fit_success"]:
            text_lines.append(
                f"{label_type}: threshold={sig['threshold_params']:.0f} params, "
                f"sharpness={sig['sharpness']:.2f}"
            )
    if text_lines:
        ax2.text(
            0.05, 0.35, "\n".join(text_lines),
            transform=ax2.transAxes, fontsize=9,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    ax2.set_xlabel("log₁₀(#parameters)", fontsize=11)
    ax2.set_ylabel("Training Accuracy", fontsize=11)
    ax2.set_title("Interpolation Threshold Comparison", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    path = os.path.join(output_dir, "threshold_comparison.png")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved {path}")
    return path
