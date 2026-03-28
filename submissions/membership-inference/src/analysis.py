"""Statistical analysis and visualization for membership inference results.

Computes correlations between attack success, model size, and overfitting gap.
Generates publication-quality plots.
"""

import json
import os
import numpy as np
from scipy import stats
from typing import Dict, List, Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def compute_correlations(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute Pearson correlations between key variables.

    Computes:
    - Attack AUC vs log(model size) [parameters]
    - Attack AUC vs overfitting gap
    - Overfitting gap vs log(model size)

    Args:
        results: List of per-width result dictionaries.

    Returns:
        Dictionary of correlation results with r-values and p-values.
    """
    widths = np.array([r["hidden_width"] for r in results])
    n_params = np.array([r["n_params"] for r in results])
    log_params = np.log2(n_params)
    auc_means = np.array([r["mean_attack_auc"] for r in results])
    gap_means = np.array([r["mean_overfit_gap"] for r in results])

    # Pearson correlations (handle degenerate case where all values are identical)
    def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> tuple:
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return (0.0, 1.0)
        return stats.pearsonr(x, y)

    r_auc_size, p_auc_size = safe_pearsonr(log_params, auc_means)
    r_auc_gap, p_auc_gap = safe_pearsonr(gap_means, auc_means)
    r_gap_size, p_gap_size = safe_pearsonr(log_params, gap_means)

    return {
        "auc_vs_log_params": {
            "r": float(r_auc_size),
            "p": float(p_auc_size),
            "description": "Attack AUC vs log2(parameter count)",
        },
        "auc_vs_overfit_gap": {
            "r": float(r_auc_gap),
            "p": float(p_auc_gap),
            "description": "Attack AUC vs overfitting gap",
        },
        "gap_vs_log_params": {
            "r": float(r_gap_size),
            "p": float(p_gap_size),
            "description": "Overfitting gap vs log2(parameter count)",
        },
    }


def generate_plots(results: List[Dict[str, Any]], output_dir: str) -> List[str]:
    """Generate publication-quality plots.

    Creates:
    1. Attack AUC vs model size (hidden width)
    2. Attack AUC vs overfitting gap (scatter with trend line)
    3. Overfitting gap vs model size

    Args:
        results: List of per-width result dictionaries.
        output_dir: Directory to save plots.

    Returns:
        List of saved plot file paths.
    """
    if not HAS_MATPLOTLIB:
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved_plots: List[str] = []

    widths = [r["hidden_width"] for r in results]
    auc_means = [r["mean_attack_auc"] for r in results]
    auc_stds = [r["std_attack_auc"] for r in results]
    gap_means = [r["mean_overfit_gap"] for r in results]
    gap_stds = [r["std_overfit_gap"] for r in results]
    n_params = [r["n_params"] for r in results]

    # Plot 1: Attack AUC vs Hidden Width
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(widths, auc_means, yerr=auc_stds, fmt="o-", capsize=4,
                linewidth=2, markersize=8, color="#2196F3")
    ax.set_xlabel("Hidden Width", fontsize=13)
    ax.set_ylabel("Attack AUC", fontsize=13)
    ax.set_title("Membership Inference Attack AUC vs Model Size", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path1 = os.path.join(output_dir, "attack_auc_vs_size.png")
    fig.savefig(path1, dpi=150)
    plt.close(fig)
    saved_plots.append(path1)

    # Plot 2: Attack AUC vs Overfitting Gap
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(gap_means, auc_means, xerr=gap_stds, yerr=auc_stds,
                fmt="o", capsize=4, markersize=10, color="#E91E63", zorder=3)

    # Add labels for each point
    for i, w in enumerate(widths):
        ax.annotate(f"w={w}", (gap_means[i], auc_means[i]),
                    textcoords="offset points", xytext=(8, 8), fontsize=10)

    # Linear fit (skip if all x values are identical)
    gap_arr = np.array(gap_means)
    auc_arr = np.array(auc_means)
    if np.std(gap_arr) > 1e-10:
        slope, intercept, r_val, p_val, _ = stats.linregress(gap_arr, auc_arr)
        x_fit = np.linspace(min(gap_arr) - 0.02, max(gap_arr) + 0.02, 50)
        y_fit = slope * x_fit + intercept
        ax.plot(x_fit, y_fit, "--", color="#9C27B0", alpha=0.7,
                label=f"Linear fit (r={r_val:.3f}, p={p_val:.3f})")

    ax.set_xlabel("Overfitting Gap (Train Acc - Test Acc)", fontsize=13)
    ax.set_ylabel("Attack AUC", fontsize=13)
    ax.set_title("Attack AUC vs Overfitting Gap", fontsize=14)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path2 = os.path.join(output_dir, "attack_auc_vs_gap.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)
    saved_plots.append(path2)

    # Plot 3: Overfitting Gap vs Hidden Width
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(widths, gap_means, yerr=gap_stds, fmt="s-", capsize=4,
                linewidth=2, markersize=8, color="#4CAF50")
    ax.set_xlabel("Hidden Width", fontsize=13)
    ax.set_ylabel("Overfitting Gap (Train Acc - Test Acc)", fontsize=13)
    ax.set_title("Overfitting Gap vs Model Size", fontsize=14)
    ax.set_xscale("log", base=2)
    ax.set_xticks(widths)
    ax.set_xticklabels([str(w) for w in widths])
    ax.axhline(y=0.0, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path3 = os.path.join(output_dir, "overfit_gap_vs_size.png")
    fig.savefig(path3, dpi=150)
    plt.close(fig)
    saved_plots.append(path3)

    # Plot 4: Combined summary (2x2 with parameter count)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 4a: AUC vs params
    ax = axes[0, 0]
    ax.errorbar(n_params, auc_means, yerr=auc_stds, fmt="o-", capsize=4,
                linewidth=2, markersize=8, color="#2196F3")
    ax.set_xlabel("Parameter Count", fontsize=12)
    ax.set_ylabel("Attack AUC", fontsize=12)
    ax.set_title("Attack AUC vs Parameter Count", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    # 4b: AUC vs gap
    ax = axes[0, 1]
    ax.errorbar(gap_means, auc_means, xerr=gap_stds, yerr=auc_stds,
                fmt="o", capsize=4, markersize=10, color="#E91E63")
    for i, w in enumerate(widths):
        ax.annotate(f"w={w}", (gap_means[i], auc_means[i]),
                    textcoords="offset points", xytext=(8, 5), fontsize=9)
    ax.set_xlabel("Overfitting Gap", fontsize=12)
    ax.set_ylabel("Attack AUC", fontsize=12)
    ax.set_title("Attack AUC vs Overfitting Gap", fontsize=13)
    ax.grid(True, alpha=0.3)

    # 4c: Gap vs params
    ax = axes[1, 0]
    ax.errorbar(n_params, gap_means, yerr=gap_stds, fmt="s-", capsize=4,
                linewidth=2, markersize=8, color="#4CAF50")
    ax.set_xlabel("Parameter Count", fontsize=12)
    ax.set_ylabel("Overfitting Gap", fontsize=12)
    ax.set_title("Overfitting Gap vs Parameter Count", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.grid(True, alpha=0.3)

    # 4d: Train/Test accuracy vs params
    ax = axes[1, 1]
    train_accs = [r["mean_train_acc"] for r in results]
    test_accs = [r["mean_test_acc"] for r in results]
    ax.plot(n_params, train_accs, "o-", label="Train Acc", linewidth=2,
            markersize=8, color="#FF9800")
    ax.plot(n_params, test_accs, "s-", label="Test Acc", linewidth=2,
            markersize=8, color="#795548")
    ax.set_xlabel("Parameter Count", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Train/Test Accuracy vs Parameter Count", fontsize=13)
    ax.set_xscale("log", base=2)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Membership Inference Scaling Analysis", fontsize=15, y=1.01)
    fig.tight_layout()
    path4 = os.path.join(output_dir, "summary_plots.png")
    fig.savefig(path4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_plots.append(path4)

    return saved_plots


def generate_report(
    results: List[Dict[str, Any]],
    correlations: Dict[str, Any],
    output_path: str,
) -> None:
    """Generate a markdown summary report.

    Args:
        results: List of per-width result dictionaries.
        correlations: Correlation analysis results.
        output_path: Path to write the report.
    """
    lines = [
        "# Membership Inference Scaling Analysis Report",
        "",
        "## Overview",
        "",
        "This experiment measures how membership inference attack success",
        "scales with MLP model size and overfitting gap, using the shadow",
        "model approach (Shokri et al., 2017).",
        "",
        "## Results by Model Size",
        "",
        "| Hidden Width | Params | Train Acc | Test Acc | Overfit Gap | Attack AUC | Attack Acc |",
        "|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]

    for r in results:
        lines.append(
            f"| {r['hidden_width']} "
            f"| {r['n_params']} "
            f"| {r['mean_train_acc']:.3f} "
            f"| {r['mean_test_acc']:.3f} "
            f"| {r['mean_overfit_gap']:.3f} +/- {r['std_overfit_gap']:.3f} "
            f"| {r['mean_attack_auc']:.3f} +/- {r['std_attack_auc']:.3f} "
            f"| {r['mean_attack_accuracy']:.3f} +/- {r['std_attack_accuracy']:.3f} |"
        )

    lines.extend([
        "",
        "## Correlation Analysis",
        "",
    ])

    for key, corr in correlations.items():
        sig = "significant" if corr["p"] < 0.05 else "not significant"
        lines.append(
            f"- **{corr['description']}**: r = {corr['r']:.4f}, "
            f"p = {corr['p']:.4f} ({sig})"
        )

    # Key findings
    auc_gap = correlations["auc_vs_overfit_gap"]
    auc_size = correlations["auc_vs_log_params"]
    gap_vs_size = correlations["gap_vs_log_params"]
    auc_gap_r = auc_gap["r"]
    auc_size_r = auc_size["r"]
    gap_is_stronger = abs(auc_gap_r) > abs(auc_size_r)

    lines.extend([
        "",
        "## Key Findings",
        "",
    ])

    if auc_gap["p"] >= 0.05 and auc_size["p"] >= 0.05:
        if gap_is_stronger:
            lines.append(
                f"- In this run, the overfitting gap has a slightly larger "
                f"correlation with attack AUC (r={auc_gap_r:.3f}, p={auc_gap['p']:.3f}) "
                f"than raw model size (r={auc_size_r:.3f}, p={auc_size['p']:.3f}), "
                "but neither association is statistically significant."
            )
        else:
            lines.append(
                f"- In this run, raw model size has a slightly larger "
                f"correlation with attack AUC (r={auc_size_r:.3f}, p={auc_size['p']:.3f}) "
                f"than the overfitting gap (r={auc_gap_r:.3f}, p={auc_gap['p']:.3f}), "
                "but neither association is statistically significant."
            )
        if gap_vs_size["p"] < 0.05:
            lines.append(
                "- The clearest supported pattern is that larger models "
                "overfit more, which may help explain the directionally higher "
                "attack AUCs without proving a predictor ranking."
            )
        else:
            lines.append(
                "- Treat the predictor ranking as directional evidence from a "
                "small toy study, not a conclusive result."
            )
    elif gap_is_stronger and auc_gap["p"] < 0.05 and auc_size["p"] >= 0.05:
        lines.append(
            f"- The overfitting gap is the better-supported predictor of attack "
            f"success in this run (r={auc_gap_r:.3f}, p={auc_gap['p']:.3f}) "
            f"than raw model size (r={auc_size_r:.3f}, p={auc_size['p']:.3f})."
        )
        lines.append(
            "- This is consistent with membership inference vulnerability being "
            "driven more by memorization behavior than by capacity alone."
        )
    elif (not gap_is_stronger) and auc_size["p"] < 0.05 and auc_gap["p"] >= 0.05:
        lines.append(
            f"- Raw model size is the better-supported predictor of attack "
            f"success in this run (r={auc_size_r:.3f}, p={auc_size['p']:.3f}) "
            f"than the overfitting gap (r={auc_gap_r:.3f}, p={auc_gap['p']:.3f})."
        )
        lines.append(
            "- This suggests capacity may matter independently of the measured "
            "generalization gap in this particular setup."
        )
    else:
        lines.append(
            f"- Both predictors show statistically supported correlations with "
            f"attack AUC, and {'the overfitting gap' if gap_is_stronger else 'raw model size'} "
            f"is numerically stronger in this run."
        )
        lines.append(
            "- Even so, the difference between the two correlations should be "
            "interpreted cautiously unless replicated at more widths."
        )

    lines.extend([
        "",
        "## Methodology",
        "",
        "- **Data**: Synthetic Gaussian clusters (500 samples, 10 features, 5 classes)",
        "- **Target models**: 2-layer MLPs with hidden widths: 16, 32, 64, 128, 256",
        "- **Shadow models**: 3 per width, same architecture, independent data",
        "- **Attack classifier**: Logistic regression on softmax prediction vectors",
        "- **Metric**: ROC AUC of membership classification",
        "- **Repeats**: 3 per width for variance estimation",
        "",
        "## Limitations",
        "",
        "- Synthetic data may not capture real-world memorization patterns.",
        "- Only five model widths are tested, which limits statistical power for the correlation analysis.",
        "- Small dataset (500 samples) amplifies overfitting even in tiny models.",
        "- Only 2-layer MLPs tested; deeper architectures may behave differently.",
        "- Shadow models use same architecture (stronger assumption than needed).",
        "",
    ])

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def save_results(
    results: List[Dict[str, Any]],
    correlations: Dict[str, Any],
    output_path: str,
) -> None:
    """Save full results to JSON.

    Args:
        results: List of per-width result dictionaries.
        correlations: Correlation analysis results.
        output_path: Path to write JSON output.
    """
    output = {
        "experiment": "membership_inference_scaling",
        "config": {
            "n_samples": 500,
            "n_features": 10,
            "n_classes": 5,
            "hidden_widths": [r["hidden_width"] for r in results],
            "n_shadow_models": 3,
            "n_repeats": 3,
            "seed": 42,
        },
        "results": results,
        "correlations": correlations,
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
