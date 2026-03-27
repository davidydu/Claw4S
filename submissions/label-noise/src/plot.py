"""Plotting utilities for label noise tolerance experiments."""

import json
import os

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


NOISE_FRACS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
NOISE_PCTS = [f"{n:.0%}" for n in NOISE_FRACS]


def load_summary(results_dir: str = "results") -> dict:
    """Load summary.json from results directory."""
    path = os.path.join(results_dir, "summary.json")
    with open(path) as f:
        return json.load(f)


def plot_arch_sweep(summary: dict, results_dir: str = "results") -> str:
    """Plot test accuracy vs noise for each architecture.

    Returns path to saved figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True)
    arch_data = summary["architecture_sweep"]

    colors = {"shallow-wide": "#2196F3", "medium": "#FF9800", "deep-narrow": "#4CAF50"}
    markers = {"shallow-wide": "o", "medium": "s", "deep-narrow": "^"}

    # Panel 1: Test accuracy vs noise
    ax = axes[0]
    for arch in arch_data:
        means = [arch_data[arch].get(nk, {}).get("test_acc_mean", 0) for nk in NOISE_PCTS]
        stds = [arch_data[arch].get(nk, {}).get("test_acc_std", 0) for nk in NOISE_PCTS]
        noise_vals = [n * 100 for n in NOISE_FRACS]
        ax.errorbar(
            noise_vals, means, yerr=stds,
            label=arch, color=colors.get(arch, "gray"),
            marker=markers.get(arch, "o"), capsize=3, linewidth=1.5,
        )
    ax.set_xlabel("Label Noise (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Test Accuracy vs Label Noise")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: Train accuracy vs noise
    ax = axes[1]
    for arch in arch_data:
        means = [arch_data[arch].get(nk, {}).get("train_acc_mean", 0) for nk in NOISE_PCTS]
        stds = [arch_data[arch].get(nk, {}).get("train_acc_std", 0) for nk in NOISE_PCTS]
        noise_vals = [n * 100 for n in NOISE_FRACS]
        ax.errorbar(
            noise_vals, means, yerr=stds,
            label=arch, color=colors.get(arch, "gray"),
            marker=markers.get(arch, "o"), capsize=3, linewidth=1.5,
        )
    ax.set_xlabel("Label Noise (%)")
    ax.set_title("Train Accuracy vs Label Noise")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Generalization gap vs noise
    ax = axes[2]
    for arch in arch_data:
        means = [arch_data[arch].get(nk, {}).get("gen_gap_mean", 0) for nk in NOISE_PCTS]
        stds = [arch_data[arch].get(nk, {}).get("gen_gap_std", 0) for nk in NOISE_PCTS]
        noise_vals = [n * 100 for n in NOISE_FRACS]
        ax.errorbar(
            noise_vals, means, yerr=stds,
            label=arch, color=colors.get(arch, "gray"),
            marker=markers.get(arch, "o"), capsize=3, linewidth=1.5,
        )
    ax.set_xlabel("Label Noise (%)")
    ax.set_title("Generalization Gap vs Label Noise")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(results_dir, "arch_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_width_sweep(summary: dict, results_dir: str = "results") -> str:
    """Plot test accuracy vs noise for width sweep (depth=2).

    Returns path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width_data = summary["width_sweep"]

    cmap = plt.cm.viridis
    widths_sorted = sorted(width_data.keys(), key=lambda w: int(w.split("_w")[1]))
    n_w = len(widths_sorted)

    # Panel 1: Test accuracy vs noise for each width
    ax = axes[0]
    for i, wname in enumerate(widths_sorted):
        width_val = int(wname.split("_w")[1])
        means = [width_data[wname].get(nk, {}).get("test_acc_mean", 0) for nk in NOISE_PCTS]
        stds = [width_data[wname].get(nk, {}).get("test_acc_std", 0) for nk in NOISE_PCTS]
        noise_vals = [n * 100 for n in NOISE_FRACS]
        ax.errorbar(
            noise_vals, means, yerr=stds,
            label=f"w={width_val}", color=cmap(i / max(n_w - 1, 1)),
            marker="o", capsize=3, linewidth=1.5,
        )
    ax.set_xlabel("Label Noise (%)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Width Effect on Noise Tolerance (depth=2)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Panel 2: Accuracy drop (0% -> 50% noise) vs width
    ax = axes[1]
    widths_int = []
    drops = []
    for wname in widths_sorted:
        width_val = int(wname.split("_w")[1])
        clean = width_data[wname].get("0%", {}).get("test_acc_mean", 0)
        noisy = width_data[wname].get("50%", {}).get("test_acc_mean", 0)
        widths_int.append(width_val)
        drops.append(clean - noisy)

    ax.bar(range(len(widths_int)), drops, color="#FF5722", alpha=0.8)
    ax.set_xticks(range(len(widths_int)))
    ax.set_xticklabels([str(w) for w in widths_int])
    ax.set_xlabel("Hidden Width")
    ax.set_ylabel("Accuracy Drop (0% -> 50% noise)")
    ax.set_title("Noise Sensitivity vs Width")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    path = os.path.join(results_dir, "width_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_all(results_dir: str = "results") -> list[str]:
    """Generate all plots. Returns list of saved file paths."""
    summary = load_summary(results_dir)
    paths = [
        plot_arch_sweep(summary, results_dir),
        plot_width_sweep(summary, results_dir),
    ]
    return paths
