"""Visualization functions for emergence analysis.

Generates publication-quality plots comparing discontinuous and continuous
metrics, showing the metric artifact mechanism, and displaying MMLU scaling.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for reproducibility
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# Plot style constants
FAMILY_COLORS = {
    "gpt3": "#1f77b4",
    "instructgpt": "#ff7f0e",
    "lamda": "#2ca02c",
    "palm": "#d62728",
    "llama": "#9467bd",
    "chinchilla": "#8c564b",
    "gopher": "#e377c2",
}

FAMILY_MARKERS = {
    "gpt3": "o",
    "instructgpt": "s",
    "lamda": "^",
    "palm": "D",
    "llama": "v",
    "chinchilla": "P",
    "gopher": "*",
}


def _setup_style():
    """Set up consistent plot style."""
    plt.rcParams.update({
        "figure.dpi": 150,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
    })


def plot_metric_comparison(comparison: dict, output_path: str) -> None:
    """Plot discontinuous vs. continuous metrics for a single task.

    Left panel: Exact match (discontinuous) vs. log(params)
    Right panel: Partial credit / per-token accuracy (continuous) vs. log(params)

    Args:
        comparison: Output from compute_metric_comparison().
        output_path: Path to save PNG.
    """
    _setup_style()
    entries = comparison["entries"]
    task_name = comparison["task"].replace("_", " ").title()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"{task_name}: Discontinuous vs. Continuous Metrics", fontsize=13)

    # Group by family for consistent colors
    families = sorted(set(e["family"] for e in entries))

    for family in families:
        family_entries = [e for e in entries if e["family"] == family]
        log_params = [np.log10(e["params_b"]) for e in family_entries]
        exact_matches = [e["exact_match"] for e in family_entries]
        partial_credits = [e["partial_credit"] for e in family_entries]

        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        label = family.upper()

        ax1.plot(log_params, exact_matches, marker=marker, color=color,
                 label=label, linewidth=1.5, markersize=7)
        ax2.plot(log_params, partial_credits, marker=marker, color=color,
                 label=label, linewidth=1.5, markersize=7)

    ax1.set_xlabel("log10(Parameters, B)")
    ax1.set_ylabel("Exact Match Accuracy")
    ax1.set_title("Discontinuous Metric\n(Exact String Match)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("log10(Parameters, B)")
    ax2.set_ylabel("Per-Token Accuracy (Partial Credit)")
    ax2.set_title("Continuous Metric\n(Inferred Per-Token Accuracy)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_synthetic_demo(demo: dict, output_path: str) -> None:
    """Plot the synthetic demonstration of the metric artifact.

    Shows how linear per-token improvement creates apparent emergence
    under exact-match scoring.

    Args:
        demo: Output from generate_synthetic_demo().
        output_path: Path to save PNG.
    """
    _setup_style()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"Metric Artifact Demonstration (n_tokens={demo['n_tokens']})",
        fontsize=13,
    )

    log_params = demo["log_params"]

    # Left: Per-token accuracy (ground truth - linear)
    ax1.plot(log_params, demo["per_token_acc"], "o-", color="#2ca02c",
             label="Per-Token Accuracy (ground truth)", markersize=5)
    ax1.plot(log_params, demo["partial_credit"], "s--", color="#1f77b4",
             label="Partial Credit (= per-token acc)", markersize=4, alpha=0.7)
    ax1.set_xlabel("log10(Parameters, B)")
    ax1.set_ylabel("Score")
    ax1.set_title("Continuous Metrics\n(smooth, predictable)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Exact match (apparent emergence)
    ax2.plot(log_params, demo["exact_match"], "o-", color="#d62728",
             label=f"Exact Match (p^{demo['n_tokens']})", markersize=5)
    ax2.plot(log_params, demo["per_token_acc"], "s--", color="#2ca02c",
             label="Per-Token Accuracy (ground truth)", markersize=4, alpha=0.5)
    ax2.set_xlabel("log10(Parameters, B)")
    ax2.set_ylabel("Score")
    ax2.set_title("Discontinuous Metric\n(apparent phase transition)")
    ax2.set_ylim(-0.05, 1.05)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add annotation about the mechanism
    ax2.annotate(
        "Same underlying\nimprovement!",
        xy=(log_params[len(log_params) // 3], demo["exact_match"][len(log_params) // 3]),
        xytext=(log_params[len(log_params) // 3] + 0.5, 0.6),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="gray",
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_nonlinearity_heatmap(scores: dict, output_path: str) -> None:
    """Plot heatmap of nonlinearity scores across tasks.

    Shows R-squared values for linear and sigmoid fits under both
    discontinuous and continuous metrics, plus MSI.

    Args:
        scores: Output from compute_nonlinearity_scores().
        output_path: Path to save PNG.
    """
    _setup_style()
    tasks = sorted(scores.keys())
    n_tasks = len(tasks)

    if n_tasks == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)
        return

    # Build data matrix
    metrics = [
        "linear_r2_discontinuous",
        "sigmoid_r2_discontinuous",
        "linear_r2_continuous",
        "sigmoid_r2_continuous",
    ]
    metric_labels = [
        "Linear R2\n(Exact Match)",
        "Sigmoid R2\n(Exact Match)",
        "Linear R2\n(Partial Credit)",
        "Sigmoid R2\n(Partial Credit)",
    ]

    data = np.zeros((n_tasks, len(metrics)))
    for i, task in enumerate(tasks):
        for j, metric in enumerate(metrics):
            data[i, j] = scores[task].get(metric, 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5),
                                    gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Nonlinearity Analysis: Discontinuous vs. Continuous Metrics",
                 fontsize=13)

    # Heatmap of R-squared values
    task_labels = [t.replace("_", " ").title() for t in tasks]
    im = ax1.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels(metric_labels, fontsize=8)
    ax1.set_yticks(range(n_tasks))
    ax1.set_yticklabels(task_labels, fontsize=8)
    ax1.set_title("R-squared Values (higher = better fit)")

    # Add text annotations
    for i in range(n_tasks):
        for j in range(len(metrics)):
            ax1.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                     fontsize=7, color="black" if data[i, j] < 0.7 else "white")

    plt.colorbar(im, ax=ax1, shrink=0.8)

    # Bar chart of MSI
    msi_values = [min(scores[t]["msi"], 10.0) for t in tasks]  # Cap for display
    colors = ["#d62728" if m > 2 else "#2ca02c" for m in msi_values]
    bars = ax2.barh(range(n_tasks), msi_values, color=colors, edgecolor="black",
                    linewidth=0.5)
    ax2.set_yticks(range(n_tasks))
    ax2.set_yticklabels(task_labels, fontsize=8)
    ax2.set_xlabel("Metric Sensitivity Index")
    ax2.set_title("MSI\n(>2 = likely artifact)")
    ax2.axvline(x=2.0, color="gray", linestyle="--", alpha=0.5, label="MSI=2 threshold")
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_mmlu_scaling(mmlu_analysis: dict, output_path: str) -> None:
    """Plot MMLU accuracy vs. model size across families.

    Shows relatively smooth scaling, contrasting with the apparent
    emergence seen in BIG-Bench exact-match tasks.

    Args:
        mmlu_analysis: Output from compute_mmlu_analysis().
        output_path: Path to save PNG.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))

    families = mmlu_analysis["families"]
    for family, data in families.items():
        log_params = [np.log10(p) for p in data["params_b"]]
        accuracies = data["accuracies"]
        color = FAMILY_COLORS.get(family, "#333333")
        marker = FAMILY_MARKERS.get(family, "o")
        label = f"{family.upper()} (lin R2={data['linear_r2']:.2f})"

        ax.plot(log_params, accuracies, marker=marker, color=color,
                label=label, linewidth=2, markersize=8)

    # Also plot single-point models
    from src.data import MMLU_DATA
    single_families = set()
    for entry in MMLU_DATA:
        fam = entry["family"]
        if fam not in families:
            single_families.add(fam)
            color = FAMILY_COLORS.get(fam, "#333333")
            marker = FAMILY_MARKERS.get(fam, "o")
            ax.plot(np.log10(entry["params_b"]), entry["accuracy"],
                    marker=marker, color=color, markersize=10,
                    label=f"{fam.upper()} ({entry['model']})")

    ax.set_xlabel("log10(Parameters, B)")
    ax.set_ylabel("5-shot MMLU Accuracy")
    ax.set_title("MMLU Scaling: Smooth Improvement Across Model Families")
    ax.set_ylim(0.2, 0.8)
    ax.axhline(y=0.25, color="gray", linestyle=":", alpha=0.5, label="Random (25%)")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Add R-squared annotation
    ax.text(
        0.98, 0.02,
        f"Overall linear R2 = {mmlu_analysis['overall_linear_r2']:.3f}",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, style="italic", color="gray",
    )

    plt.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
