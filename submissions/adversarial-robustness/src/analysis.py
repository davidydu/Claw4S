"""
Analysis and visualization for adversarial robustness scaling experiments.

Produces plots and summary statistics from experiment results.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def compute_robustness_gaps(results: list[dict]) -> list[dict]:
    """Compute the robustness gap (clean_acc - robust_acc) for each result.

    Args:
        results: List of experiment result dictionaries, each containing
                 'hidden_width', 'clean_acc', 'fgsm_acc', 'pgd_acc', 'epsilon'.

    Returns:
        List of dicts augmented with 'fgsm_gap' and 'pgd_gap' fields.
    """
    augmented = []
    for r in results:
        entry = dict(r)
        entry["fgsm_gap"] = r["clean_acc"] - r["fgsm_acc"]
        entry["pgd_gap"] = r["clean_acc"] - r["pgd_acc"]
        augmented.append(entry)
    return augmented


def compute_summary_statistics(results: list[dict]) -> dict:
    """Compute summary statistics across all experiments.

    Args:
        results: List of experiment result dicts with robustness gaps.

    Returns:
        Dictionary with summary statistics.
    """
    widths = sorted(set(r["hidden_width"] for r in results))
    epsilons = sorted(set(r["epsilon"] for r in results))

    summary = {
        "widths": widths,
        "epsilons": epsilons,
        "n_experiments": len(results),
    }

    # Per-width averages
    per_width = {}
    for w in widths:
        w_results = [r for r in results if r["hidden_width"] == w]
        per_width[w] = {
            "clean_acc": w_results[0]["clean_acc"],  # same for all epsilons
            "param_count": w_results[0]["param_count"],
            "mean_fgsm_gap": float(np.mean([r["fgsm_gap"] for r in w_results])),
            "mean_pgd_gap": float(np.mean([r["pgd_gap"] for r in w_results])),
            "mean_fgsm_acc": float(np.mean([r["fgsm_acc"] for r in w_results])),
            "mean_pgd_acc": float(np.mean([r["pgd_acc"] for r in w_results])),
            "std_fgsm_gap": float(np.std([r["fgsm_gap"] for r in w_results])),
            "std_pgd_gap": float(np.std([r["pgd_gap"] for r in w_results])),
        }
    summary["per_width"] = per_width

    # Correlation: param_count vs mean_gap
    param_counts = np.array([per_width[w]["param_count"] for w in widths], dtype=float)
    mean_fgsm_gaps = np.array([per_width[w]["mean_fgsm_gap"] for w in widths])
    mean_pgd_gaps = np.array([per_width[w]["mean_pgd_gap"] for w in widths])

    if len(widths) >= 3:
        log_params = np.log10(param_counts)
        summary["corr_logparams_fgsm_gap"] = float(np.corrcoef(log_params, mean_fgsm_gaps)[0, 1])
        summary["corr_logparams_pgd_gap"] = float(np.corrcoef(log_params, mean_pgd_gaps)[0, 1])
    else:
        summary["corr_logparams_fgsm_gap"] = None
        summary["corr_logparams_pgd_gap"] = None

    return summary


def plot_clean_vs_robust(results: list[dict], output_dir: str) -> str:
    """Plot clean accuracy and robust accuracy vs model size.

    Args:
        results: List of experiment result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot.
    """
    widths = sorted(set(r["hidden_width"] for r in results))
    epsilons = sorted(set(r["epsilon"] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FGSM plot
    ax = axes[0]
    clean_accs = [results[[r["hidden_width"] for r in results].index(w)]["clean_acc"]
                  for w in widths]
    ax.plot(widths, clean_accs, "k-o", linewidth=2, label="Clean", markersize=6)
    for eps in epsilons:
        accs = [r["fgsm_acc"] for r in results
                if r["epsilon"] == eps]
        ax.plot(widths, accs, "--s", label=f"FGSM eps={eps}", markersize=4)
    ax.set_xlabel("Hidden Width", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("FGSM: Clean vs Robust Accuracy", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # PGD plot
    ax = axes[1]
    ax.plot(widths, clean_accs, "k-o", linewidth=2, label="Clean", markersize=6)
    for eps in epsilons:
        accs = [r["pgd_acc"] for r in results
                if r["epsilon"] == eps]
        ax.plot(widths, accs, "--s", label=f"PGD eps={eps}", markersize=4)
    ax.set_xlabel("Hidden Width", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("PGD: Clean vs Robust Accuracy", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    plt.tight_layout()
    path = os.path.join(output_dir, "clean_vs_robust.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_robustness_gap(results: list[dict], output_dir: str) -> str:
    """Plot robustness gap (clean - robust) vs model size.

    Args:
        results: List of experiment result dicts with gap fields.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot.
    """
    widths = sorted(set(r["hidden_width"] for r in results))
    epsilons = sorted(set(r["epsilon"] for r in results))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # FGSM gaps
    ax = axes[0]
    for eps in epsilons:
        gaps = [r["fgsm_gap"] for r in results if r["epsilon"] == eps]
        ax.plot(widths, gaps, "-o", label=f"eps={eps}", markersize=5)
    ax.set_xlabel("Hidden Width", fontsize=12)
    ax.set_ylabel("Robustness Gap (Clean - FGSM)", fontsize=12)
    ax.set_title("FGSM Robustness Gap vs Model Size", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # PGD gaps
    ax = axes[1]
    for eps in epsilons:
        gaps = [r["pgd_gap"] for r in results if r["epsilon"] == eps]
        ax.plot(widths, gaps, "-o", label=f"eps={eps}", markersize=5)
    ax.set_xlabel("Hidden Width", fontsize=12)
    ax.set_ylabel("Robustness Gap (Clean - PGD)", fontsize=12)
    ax.set_title("PGD Robustness Gap vs Model Size", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "robustness_gap.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_param_count_scaling(results: list[dict], output_dir: str) -> str:
    """Plot robustness metrics vs parameter count (log-log).

    Args:
        results: List of experiment result dicts.
        output_dir: Directory to save the plot.

    Returns:
        Path to the saved plot.
    """
    widths = sorted(set(r["hidden_width"] for r in results))

    # Aggregate per width
    per_width = {}
    for w in widths:
        w_results = [r for r in results if r["hidden_width"] == w]
        per_width[w] = {
            "param_count": w_results[0]["param_count"],
            "clean_acc": w_results[0]["clean_acc"],
            "mean_fgsm_acc": float(np.mean([r["fgsm_acc"] for r in w_results])),
            "mean_pgd_acc": float(np.mean([r["pgd_acc"] for r in w_results])),
            "mean_fgsm_gap": float(np.mean([r["fgsm_gap"] for r in w_results])),
            "mean_pgd_gap": float(np.mean([r["pgd_gap"] for r in w_results])),
        }

    param_counts = [per_width[w]["param_count"] for w in widths]
    mean_fgsm_gaps = [per_width[w]["mean_fgsm_gap"] for w in widths]
    mean_pgd_gaps = [per_width[w]["mean_pgd_gap"] for w in widths]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(param_counts, mean_fgsm_gaps, "b-o", label="Mean FGSM Gap", markersize=7)
    ax.plot(param_counts, mean_pgd_gaps, "r-s", label="Mean PGD Gap", markersize=7)
    ax.set_xlabel("Parameter Count", fontsize=12)
    ax.set_ylabel("Mean Robustness Gap", fontsize=12)
    ax.set_title("Robustness Gap Scaling with Model Size", fontsize=13)
    ax.set_xscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "param_scaling.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_results(results: list[dict], summary: dict, output_dir: str) -> str:
    """Save results and summary to JSON.

    Args:
        results: List of experiment result dicts.
        summary: Summary statistics dict.
        output_dir: Directory to save the JSON file.

    Returns:
        Path to the saved JSON file.
    """
    output = {
        "results": results,
        "summary": summary,
    }
    path = os.path.join(output_dir, "results.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path
