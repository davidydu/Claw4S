"""Analysis and plotting for lottery ticket experiments.

Generates accuracy-vs-sparsity curves, strategy comparisons,
and summary statistics.
"""

import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_results(results_path: str = "results/results.json") -> dict:
    """Load experiment results from JSON file."""
    with open(results_path) as f:
        return json.load(f)


def compute_summary_stats(results: list[dict]) -> dict:
    """Compute mean and std of metrics grouped by (task, strategy, sparsity).

    Returns:
        Nested dict: summary[task][strategy][sparsity] = {metric_mean, metric_std, ...}
    """
    groups = defaultdict(list)
    for r in results:
        key = (r["task"], r["strategy"], r["sparsity"])
        groups[key].append(r)

    summary = {}
    for (task, strategy, sparsity), runs in groups.items():
        if task not in summary:
            summary[task] = {}
        if strategy not in summary[task]:
            summary[task][strategy] = {}

        if task == "modular":
            metric_key = "test_acc"
        else:
            metric_key = "test_r2"

        values = [r[metric_key] for r in runs]
        epochs = [r["epochs_trained"] for r in runs]

        summary[task][strategy][sparsity] = {
            "metric_mean": float(np.mean(values)),
            "metric_std": float(np.std(values)),
            "metric_values": values,
            "epochs_mean": float(np.mean(epochs)),
            "epochs_std": float(np.std(epochs)),
            "n_seeds": len(runs),
        }

    return summary


def find_critical_sparsity(summary: dict, task: str, strategy: str, threshold: float = 0.95) -> float:
    """Find the highest sparsity where performance stays above threshold * dense performance.

    Args:
        summary: Output of compute_summary_stats.
        task: "modular" or "regression".
        strategy: "magnitude", "random", or "structured".
        threshold: Fraction of dense performance to maintain (default 95%).

    Returns:
        Critical sparsity level, or 0.0 if no data.
    """
    if task not in summary or strategy not in summary[task]:
        return 0.0

    data = summary[task][strategy]
    if 0.0 not in data:
        return 0.0

    dense_perf = data[0.0]["metric_mean"]
    target = threshold * dense_perf

    critical = 0.0
    for sparsity in sorted(data.keys()):
        if data[sparsity]["metric_mean"] >= target:
            critical = sparsity

    return critical


def plot_accuracy_vs_sparsity(summary: dict, output_dir: str = "results") -> str:
    """Plot accuracy/R2 vs sparsity for all strategies, one subplot per task.

    Returns:
        Path to saved figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    task_labels = {"modular": "Modular Arithmetic (mod 97)", "regression": "Regression (n=200, d=20)"}
    metric_labels = {"modular": "Test Accuracy", "regression": "Test R$^2$"}
    strategy_colors = {"magnitude": "#2196F3", "random": "#FF9800", "structured": "#4CAF50"}
    strategy_markers = {"magnitude": "o", "random": "s", "structured": "^"}

    for idx, task in enumerate(["modular", "regression"]):
        ax = axes[idx]
        if task not in summary:
            continue

        for strategy in ["magnitude", "random", "structured"]:
            if strategy not in summary[task]:
                continue

            data = summary[task][strategy]
            sparsities = sorted(data.keys())
            means = [data[s]["metric_mean"] for s in sparsities]
            stds = [data[s]["metric_std"] for s in sparsities]

            ax.errorbar(
                [s * 100 for s in sparsities],
                means,
                yerr=stds,
                label=strategy.capitalize(),
                color=strategy_colors[strategy],
                marker=strategy_markers[strategy],
                markersize=7,
                linewidth=2,
                capsize=4,
            )

        # Mark critical sparsity for magnitude pruning
        crit = find_critical_sparsity(summary, task, "magnitude")
        if crit > 0:
            ax.axvline(crit * 100, color="#E91E63", linestyle="--", alpha=0.7,
                       label=f"Critical sparsity ({crit:.0%})")

        ax.set_xlabel("Sparsity (%)", fontsize=12)
        ax.set_ylabel(metric_labels.get(task, "Metric"), fontsize=12)
        ax.set_title(task_labels.get(task, task), fontsize=13)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5, 100)

    fig.suptitle("Lottery Tickets at Birth: Accuracy vs. Sparsity", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "accuracy_vs_sparsity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")
    return path


def plot_epochs_vs_sparsity(summary: dict, output_dir: str = "results") -> str:
    """Plot training epochs vs sparsity for magnitude pruning."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    task_labels = {"modular": "Modular Arithmetic", "regression": "Regression"}

    for idx, task in enumerate(["modular", "regression"]):
        ax = axes[idx]
        if task not in summary:
            continue

        strategy = "magnitude"
        if strategy not in summary[task]:
            continue

        data = summary[task][strategy]
        sparsities = sorted(data.keys())
        means = [data[s]["epochs_mean"] for s in sparsities]
        stds = [data[s]["epochs_std"] for s in sparsities]

        ax.bar(
            [s * 100 for s in sparsities],
            means,
            yerr=stds,
            width=6,
            color="#2196F3",
            alpha=0.8,
            capsize=4,
        )

        ax.set_xlabel("Sparsity (%)", fontsize=12)
        ax.set_ylabel("Epochs to Convergence", fontsize=12)
        ax.set_title(task_labels.get(task, task), fontsize=13)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Training Duration vs. Sparsity (Magnitude Pruning)", fontsize=15, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(output_dir, "epochs_vs_sparsity.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {path}")
    return path


def generate_report(results_data: dict, output_dir: str = "results") -> str:
    """Generate a text report summarizing the experiments.

    Args:
        results_data: Full results dictionary from run_all_experiments.
        output_dir: Directory to save the report.

    Returns:
        Report text.
    """
    summary = compute_summary_stats(results_data["results"])
    meta = results_data["metadata"]

    lines = []
    lines.append("=" * 70)
    lines.append("LOTTERY TICKETS AT BIRTH: EXPERIMENT REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Total runs: {meta['total_runs']}")
    lines.append(f"Runtime: {meta['elapsed_seconds']}s")
    lines.append(f"Model: 2-layer MLP, hidden_dim={meta['hidden_dim']}")
    lines.append(f"Seeds: {len(results_data['results']) // (meta['num_tasks'] * meta['num_strategies'] * meta['num_sparsity_levels'])}")
    lines.append("")

    for task in ["modular", "regression"]:
        metric_name = "Test Accuracy" if task == "modular" else "Test R^2"
        lines.append("-" * 50)
        lines.append(f"Task: {task.upper()}")
        lines.append("-" * 50)

        for strategy in ["magnitude", "random", "structured"]:
            if task not in summary or strategy not in summary[task]:
                continue

            data = summary[task][strategy]
            lines.append(f"\n  Strategy: {strategy.capitalize()}")
            lines.append(f"  {'Sparsity':>10s}  {metric_name:>15s}  {'Std':>8s}  {'Epochs':>8s}")

            for sparsity in sorted(data.keys()):
                s = data[sparsity]
                lines.append(
                    f"  {sparsity:>9.0%}  {s['metric_mean']:>15.4f}  "
                    f"{s['metric_std']:>8.4f}  {s['epochs_mean']:>8.0f}"
                )

            crit = find_critical_sparsity(summary, task, strategy)
            lines.append(f"  Critical sparsity (95% threshold): {crit:.0%}")

    # Key findings
    lines.append("")
    lines.append("=" * 70)
    lines.append("KEY FINDINGS")
    lines.append("=" * 70)

    for task in ["modular", "regression"]:
        if task not in summary:
            continue

        metric_name = "accuracy" if task == "modular" else "R^2"
        lines.append(f"\n{task.upper()}:")

        # Compare strategies at high sparsity
        for sparsity in [0.7, 0.9]:
            lines.append(f"  At {sparsity:.0%} sparsity:")
            for strategy in ["magnitude", "random", "structured"]:
                if strategy in summary[task] and sparsity in summary[task][strategy]:
                    val = summary[task][strategy][sparsity]["metric_mean"]
                    lines.append(f"    {strategy:>12s}: {metric_name}={val:.4f}")

        # Critical sparsity comparison
        lines.append("  Critical sparsity (95% of dense performance):")
        for strategy in ["magnitude", "random", "structured"]:
            crit = find_critical_sparsity(summary, task, strategy)
            lines.append(f"    {strategy:>12s}: {crit:.0%}")

    report = "\n".join(lines)

    report_path = os.path.join(output_dir, "report.txt")
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")

    return report
