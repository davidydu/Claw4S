"""Visualization functions for Benford's Law analysis results."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.benford_analysis import benford_expected


def plot_digit_distribution(observed, title, save_path, expected=None):
    """Plot observed digit distribution with Benford's Law overlay.

    Args:
        observed: Dict mapping digit (str or int) to observed proportion.
        title: Plot title.
        save_path: File path to save the figure.
        expected: Optional expected distribution dict. Defaults to Benford's.
    """
    if expected is None:
        expected = benford_expected()

    digits = list(range(1, 10))
    obs_vals = [observed.get(str(d), observed.get(d, 0)) for d in digits]
    exp_vals = [expected[d] for d in digits]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(digits))
    width = 0.35

    bars_obs = ax.bar(x - width / 2, obs_vals, width, label="Observed", color="#4C72B0")
    bars_exp = ax.bar(x + width / 2, exp_vals, width, label="Benford Expected", color="#DD8452")

    ax.set_xlabel("Leading Digit")
    ax.set_ylabel("Proportion")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(d) for d in digits])
    ax.legend()
    ax.set_ylim(0, max(max(obs_vals), max(exp_vals)) * 1.2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mad_over_training(results_by_epoch, title, save_path):
    """Plot MAD from Benford's Law over training epochs.

    Args:
        results_by_epoch: Dict mapping epoch (int) to analysis result dict
            containing "mad" key.
        title: Plot title.
        save_path: File path to save the figure.
    """
    epochs = sorted(results_by_epoch.keys())
    mads = [results_by_epoch[e]["mad"] for e in epochs]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(epochs, mads, "o-", color="#4C72B0", linewidth=2, markersize=8)

    # Add Nigrini threshold lines
    ax.axhline(y=0.006, color="green", linestyle="--", alpha=0.7, label="Close (0.006)")
    ax.axhline(y=0.012, color="orange", linestyle="--", alpha=0.7, label="Acceptable (0.012)")
    ax.axhline(y=0.015, color="red", linestyle="--", alpha=0.7, label="Marginal (0.015)")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MAD from Benford")
    ax.set_title(title)
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_layer_comparison(per_layer_results, title, save_path):
    """Plot MAD values comparing different layers.

    Args:
        per_layer_results: Dict mapping layer name to analysis result dict.
        title: Plot title.
        save_path: File path to save the figure.
    """
    layers = sorted(per_layer_results.keys())
    mads = [per_layer_results[l]["mad"] for l in layers]

    # Shorten layer names for display
    short_names = []
    for l in layers:
        parts = l.replace("net.", "").replace(".weight", "")
        short_names.append(f"Layer {parts}")

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]
    bar_colors = [colors[i % len(colors)] for i in range(len(layers))]

    ax.bar(range(len(layers)), mads, color=bar_colors)

    # Add threshold lines
    ax.axhline(y=0.006, color="green", linestyle="--", alpha=0.7, label="Close (0.006)")
    ax.axhline(y=0.012, color="orange", linestyle="--", alpha=0.7, label="Acceptable (0.012)")
    ax.axhline(y=0.015, color="red", linestyle="--", alpha=0.7, label="Marginal (0.015)")

    ax.set_xlabel("Layer")
    ax.set_ylabel("MAD from Benford")
    ax.set_title(title)
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_controls_comparison(controls, save_path):
    """Plot MAD comparison across control distributions and Nigrini thresholds.

    Args:
        controls: Dict mapping control name to analysis result dict.
        save_path: File path to save the figure.
    """
    names = list(controls.keys())
    mads = [controls[n]["mad"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    bar_colors = [colors[i % len(colors)] for i in range(len(names))]

    ax.bar(range(len(names)), mads, color=bar_colors)

    ax.axhline(y=0.006, color="green", linestyle="--", alpha=0.7, label="Close (0.006)")
    ax.axhline(y=0.012, color="orange", linestyle="--", alpha=0.7, label="Acceptable (0.012)")
    ax.axhline(y=0.015, color="red", linestyle="--", alpha=0.7, label="Marginal (0.015)")

    ax.set_xlabel("Distribution")
    ax.set_ylabel("MAD from Benford")
    ax.set_title("Control Distributions: MAD from Benford's Law")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("_", " ").title() for n in names])
    ax.legend(loc="upper right")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
