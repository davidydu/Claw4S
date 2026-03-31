"""Visualization for membership inference under DP experiments.

Generates three key plots:
  1. Attack AUC vs. privacy level (the main result)
  2. Privacy-utility-leakage triad
  3. Generalization gap vs. attack success
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_attack_auc_vs_privacy(results: dict, output_dir: str) -> str:
    """Bar chart of attack AUC across privacy levels.

    Shows mean +/- std of membership inference AUC for each
    privacy configuration, with a dashed line at AUC=0.5 (random).

    Args:
        results: Full experiment results.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot.
    """
    agg = results["aggregated"]
    levels = list(agg.keys())
    means = [agg[l]["metrics"]["attack_auc"]["mean"] for l in levels]
    stds = [agg[l]["metrics"]["attack_auc"]["std"] for l in levels]
    sigmas = [agg[l]["noise_multiplier"] for l in levels]

    labels = [f"{l}\n(sigma={s})" for l, s in zip(levels, sigmas)]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#d32f2f", "#ff9800", "#2196f3", "#4caf50"]
    bars = ax.bar(range(len(levels)), means, yerr=stds, capsize=5,
                  color=colors[:len(levels)], edgecolor="black", linewidth=0.5)

    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Attack AUC", fontsize=12)
    ax.set_title("Membership Inference Attack AUC vs. Privacy Level", fontsize=13)
    ax.set_ylim(0.3, 1.0)
    ax.legend(fontsize=10)

    # Add value labels on bars
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.02,
                f"{m:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(output_dir, "attack_auc_vs_privacy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_privacy_utility_leakage(results: dict, output_dir: str) -> str:
    """Three-panel plot showing the privacy-utility-leakage triad.

    Left: Test accuracy (utility) vs. noise multiplier
    Center: Attack AUC (leakage) vs. noise multiplier
    Right: Utility vs. Leakage trade-off

    Args:
        results: Full experiment results.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot.
    """
    agg = results["aggregated"]
    levels = list(agg.keys())
    sigmas = [agg[l]["noise_multiplier"] for l in levels]
    test_accs = [agg[l]["metrics"]["test_accuracy"]["mean"] for l in levels]
    test_stds = [agg[l]["metrics"]["test_accuracy"]["std"] for l in levels]
    attack_aucs = [agg[l]["metrics"]["attack_auc"]["mean"] for l in levels]
    attack_stds = [agg[l]["metrics"]["attack_auc"]["std"] for l in levels]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Left: Utility vs sigma
    axes[0].errorbar(sigmas, test_accs, yerr=test_stds, marker="o", capsize=4,
                     color="#2196f3", linewidth=2)
    axes[0].set_xlabel("Noise Multiplier (sigma)", fontsize=11)
    axes[0].set_ylabel("Test Accuracy", fontsize=11)
    axes[0].set_title("Model Utility", fontsize=12)
    axes[0].set_ylim(0, 1)

    # Center: Leakage vs sigma
    axes[1].errorbar(sigmas, attack_aucs, yerr=attack_stds, marker="s", capsize=4,
                     color="#d32f2f", linewidth=2)
    axes[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.7)
    axes[1].set_xlabel("Noise Multiplier (sigma)", fontsize=11)
    axes[1].set_ylabel("Attack AUC", fontsize=11)
    axes[1].set_title("Privacy Leakage", fontsize=12)
    axes[1].set_ylim(0.3, 1.0)

    # Right: Utility vs Leakage
    colors = ["#d32f2f", "#ff9800", "#2196f3", "#4caf50"]
    for i, (l, ta, aa) in enumerate(zip(levels, test_accs, attack_aucs)):
        axes[2].scatter(aa, ta, s=100, color=colors[i], zorder=5, label=l)
    axes[2].set_xlabel("Attack AUC (Leakage)", fontsize=11)
    axes[2].set_ylabel("Test Accuracy (Utility)", fontsize=11)
    axes[2].set_title("Privacy-Utility Trade-off", fontsize=12)
    axes[2].axvline(x=0.5, color="gray", linestyle="--", alpha=0.7)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(output_dir, "privacy_utility_leakage.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_generalization_gap(results: dict, output_dir: str) -> str:
    """Scatter plot of generalization gap vs. attack AUC.

    Hypothesis: models that overfit more (larger gap) are more
    vulnerable to membership inference.

    Args:
        results: Full experiment results.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot.
    """
    per_trial = results["per_trial"]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors_map = {
        "non-private": "#d32f2f",
        "weak-dp": "#ff9800",
        "moderate-dp": "#2196f3",
        "strong-dp": "#4caf50",
    }

    for trial in per_trial:
        ax.scatter(
            trial["generalization_gap"],
            trial["attack_auc"],
            color=colors_map.get(trial["privacy_level"], "gray"),
            s=60, alpha=0.8,
            label=trial["privacy_level"],
        )

    # Deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=9)

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="_random")
    ax.set_xlabel("Generalization Gap (Train Acc - Test Acc)", fontsize=11)
    ax.set_ylabel("Attack AUC", fontsize=11)
    ax.set_title("Overfitting Correlates with Membership Leakage", fontsize=12)

    plt.tight_layout()
    path = os.path.join(output_dir, "generalization_gap_vs_attack.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_all_plots(results: dict, output_dir: str) -> list[str]:
    """Generate all plots for the experiment.

    Args:
        results: Full experiment results.
        output_dir: Directory to save plots.

    Returns:
        List of paths to generated plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    paths = [
        plot_attack_auc_vs_privacy(results, output_dir),
        plot_privacy_utility_leakage(results, output_dir),
        plot_generalization_gap(results, output_dir),
    ]

    print(f"Generated {len(paths)} plots in {output_dir}/")
    for p in paths:
        print(f"  - {os.path.basename(p)}")

    return paths
