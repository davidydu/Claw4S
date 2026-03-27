"""Analysis and visualization for DP-SGD privacy-utility tradeoff.

Generates plots and summary statistics from experiment results.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(results_dir: str) -> dict:
    """Load experiment results from JSON file.

    Args:
        results_dir: Path to results directory.

    Returns:
        Dictionary with experiment results.
    """
    path = os.path.join(results_dir, "results.json")
    with open(path, "r") as f:
        return json.load(f)


def compute_summary_statistics(results: dict) -> dict:
    """Compute summary statistics across seeds and configurations.

    Groups results by (noise_multiplier, max_norm), computes mean and
    std of accuracy and epsilon across seeds.

    Args:
        results: Raw experiment results dictionary.

    Returns:
        Dictionary with summary statistics per configuration.
    """
    dp_runs = results["dp_runs"]
    baseline = results["baseline_runs"]

    # Baseline accuracy (mean across seeds)
    baseline_accs = [r["accuracy"] for r in baseline]
    baseline_mean = float(np.mean(baseline_accs))
    baseline_std = float(np.std(baseline_accs))

    # Group DP runs by (noise_multiplier, max_norm)
    groups: dict[tuple[float, float], list[dict]] = {}
    for run in dp_runs:
        key = (run["noise_multiplier"], run["max_norm"])
        groups.setdefault(key, []).append(run)

    summaries = []
    for (sigma, C), runs in sorted(groups.items()):
        accs = [r["accuracy"] for r in runs]
        epsilons = [r["epsilon"] for r in runs]
        utility_gaps = [r["accuracy"] - baseline_mean for r in runs]

        summaries.append({
            "noise_multiplier": sigma,
            "max_norm": C,
            "accuracy_mean": float(np.mean(accs)),
            "accuracy_std": float(np.std(accs)),
            "epsilon_mean": float(np.mean(epsilons)),
            "epsilon_std": float(np.std(epsilons)),
            "utility_gap_mean": float(np.mean(utility_gaps)),
            "utility_gap_std": float(np.std(utility_gaps)),
            "n_seeds": len(runs),
        })

    return {
        "baseline_accuracy_mean": baseline_mean,
        "baseline_accuracy_std": baseline_std,
        "configurations": summaries,
    }


def identify_privacy_cliff(summaries: dict, threshold_fraction: float = 0.5) -> dict:
    """Identify the privacy cliff — the epsilon below which utility collapses.

    The "cliff" is defined as the epsilon where accuracy drops below
    threshold_fraction of the baseline accuracy.

    Args:
        summaries: Output of compute_summary_statistics.
        threshold_fraction: Fraction of baseline accuracy below which
            utility is considered collapsed.

    Returns:
        Dictionary describing the privacy cliff finding.
    """
    baseline_acc = summaries["baseline_accuracy_mean"]
    threshold = baseline_acc * threshold_fraction
    configs = summaries["configurations"]

    # Sort by epsilon (ascending)
    sorted_configs = sorted(configs, key=lambda c: c["epsilon_mean"])

    cliff_epsilon = None
    cliff_accuracy = None

    # Find the first config above threshold (going from low to high epsilon)
    for i, cfg in enumerate(sorted_configs):
        if cfg["accuracy_mean"] >= threshold:
            cliff_epsilon = cfg["epsilon_mean"]
            cliff_accuracy = cfg["accuracy_mean"]
            # The cliff is between this config and the previous one
            if i > 0:
                below = sorted_configs[i - 1]
                cliff_epsilon = (below["epsilon_mean"] + cfg["epsilon_mean"]) / 2
            break

    # Identify "safe" region: where accuracy is >= 90% of baseline
    safe_threshold = baseline_acc * 0.9
    safe_configs = [c for c in sorted_configs if c["accuracy_mean"] >= safe_threshold]
    safe_epsilon = min(c["epsilon_mean"] for c in safe_configs) if safe_configs else None

    return {
        "cliff_epsilon": cliff_epsilon,
        "cliff_accuracy": cliff_accuracy,
        "threshold_fraction": threshold_fraction,
        "baseline_accuracy": baseline_acc,
        "safe_epsilon": safe_epsilon,
        "safe_threshold_fraction": 0.9,
        "n_configs_below_threshold": sum(
            1 for c in sorted_configs if c["accuracy_mean"] < threshold
        ),
        "n_configs_total": len(sorted_configs),
    }


def plot_privacy_utility_curve(
    summaries: dict,
    output_path: str,
    max_norm_filter: float | None = None,
) -> None:
    """Plot accuracy vs epsilon (privacy-utility tradeoff).

    Args:
        summaries: Output of compute_summary_statistics.
        output_path: Path to save the plot.
        max_norm_filter: If set, only plot configs with this max_norm.
    """
    configs = summaries["configurations"]
    baseline_acc = summaries["baseline_accuracy_mean"]

    if max_norm_filter is not None:
        configs = [c for c in configs if c["max_norm"] == max_norm_filter]

    # Group by max_norm
    norms = sorted(set(c["max_norm"] for c in configs))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for C in norms:
        subset = sorted(
            [c for c in configs if c["max_norm"] == C],
            key=lambda c: c["epsilon_mean"],
        )
        epsilons = [c["epsilon_mean"] for c in subset]
        accs = [c["accuracy_mean"] for c in subset]
        stds = [c["accuracy_std"] for c in subset]

        ax.errorbar(
            epsilons, accs, yerr=stds,
            marker="o", label=f"C={C}", capsize=3, linewidth=1.5,
        )

    # Baseline
    ax.axhline(
        y=baseline_acc, color="black", linestyle="--", linewidth=1,
        label=f"Non-private baseline ({baseline_acc:.3f})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("DP-SGD Privacy-Utility Tradeoff", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_utility_gap(
    summaries: dict,
    output_path: str,
) -> None:
    """Plot utility gap (private - baseline accuracy) vs epsilon.

    Args:
        summaries: Output of compute_summary_statistics.
        output_path: Path to save the plot.
    """
    configs = summaries["configurations"]
    norms = sorted(set(c["max_norm"] for c in configs))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for C in norms:
        subset = sorted(
            [c for c in configs if c["max_norm"] == C],
            key=lambda c: c["epsilon_mean"],
        )
        epsilons = [c["epsilon_mean"] for c in subset]
        gaps = [c["utility_gap_mean"] for c in subset]
        stds = [c["utility_gap_std"] for c in subset]

        ax.errorbar(
            epsilons, gaps, yerr=stds,
            marker="s", label=f"C={C}", capsize=3, linewidth=1.5,
        )

    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, label="Zero gap")
    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget (epsilon)", fontsize=12)
    ax.set_ylabel("Utility Gap (private - baseline)", fontsize=12)
    ax.set_title("Utility Gap vs Privacy Budget", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_clipping_effect(
    summaries: dict,
    output_path: str,
) -> None:
    """Plot accuracy vs clipping norm for each noise level.

    Shows how clipping norm C interacts with privacy noise.

    Args:
        summaries: Output of compute_summary_statistics.
        output_path: Path to save the plot.
    """
    configs = summaries["configurations"]
    sigmas = sorted(set(c["noise_multiplier"] for c in configs))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    for sigma in sigmas:
        subset = sorted(
            [c for c in configs if c["noise_multiplier"] == sigma],
            key=lambda c: c["max_norm"],
        )
        norms = [c["max_norm"] for c in subset]
        accs = [c["accuracy_mean"] for c in subset]
        stds = [c["accuracy_std"] for c in subset]

        ax.errorbar(
            norms, accs, yerr=stds,
            marker="^", label=f"sigma={sigma}", capsize=3, linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Clipping Norm (C)", fontsize=12)
    ax.set_ylabel("Test Accuracy", fontsize=12)
    ax.set_title("Effect of Gradient Clipping on Accuracy", fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_plots(results_dir: str) -> list[str]:
    """Generate all analysis plots from results.

    Args:
        results_dir: Path to results directory.

    Returns:
        List of paths to generated plot files.
    """
    results = load_results(results_dir)
    summaries = compute_summary_statistics(results)

    plots = []

    # Privacy-utility curve
    path1 = os.path.join(results_dir, "privacy_utility_curve.png")
    plot_privacy_utility_curve(summaries, path1)
    plots.append(path1)

    # Utility gap
    path2 = os.path.join(results_dir, "utility_gap.png")
    plot_utility_gap(summaries, path2)
    plots.append(path2)

    # Clipping effect
    path3 = os.path.join(results_dir, "clipping_effect.png")
    plot_clipping_effect(summaries, path3)
    plots.append(path3)

    # Save summary
    cliff = identify_privacy_cliff(summaries)
    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {"summaries": summaries, "privacy_cliff": cliff},
            f, indent=2,
        )
    plots.append(summary_path)

    return plots
