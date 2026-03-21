"""Plotting module for benchmark difficulty prediction.

Generates three figures:
  1. Feature correlation bar chart (Spearman rho per feature)
  2. Predicted vs actual difficulty scatter plot
  3. Feature importance bar chart (Random Forest importances)
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_feature_correlations(
    correlations: dict[str, dict],
    output_path: str,
) -> None:
    """Plot Spearman correlations between features and difficulty.

    Args:
        correlations: Dict mapping feature name -> {"rho": float, "pvalue": float}.
        output_path: Path to save the figure.
    """
    # Sort by absolute rho
    sorted_items = sorted(
        correlations.items(),
        key=lambda x: abs(x[1]["rho"]),
        reverse=True,
    )
    names = [item[0] for item in sorted_items]
    rhos = [item[1]["rho"] for item in sorted_items]
    pvals = [item[1]["pvalue"] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71" if p < 0.05 else "#95a5a6" for p in pvals]
    bars = ax.barh(range(len(names)), rhos, color=colors, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Spearman Correlation (rho)", fontsize=12)
    ax.set_title("Feature Correlations with IRT Difficulty", fontsize=14)
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ecc71", label="p < 0.05"),
        Patch(facecolor="#95a5a6", label="p >= 0.05"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_difficulty_prediction(
    predictions: list[float],
    actual: list[float],
    output_path: str,
) -> None:
    """Plot predicted vs actual difficulty scatter plot.

    Args:
        predictions: Model predictions.
        actual: Ground-truth difficulty scores.
        output_path: Path to save the figure.
    """
    pred = np.array(predictions)
    act = np.array(actual)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(act, pred, alpha=0.6, s=30, color="#3498db", edgecolors="white",
               linewidth=0.5)

    # Diagonal line (perfect prediction)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect prediction")

    # Regression line
    from scipy import stats
    slope, intercept, r_value, p_value, std_err = stats.linregress(act, pred)
    x_line = np.linspace(0, 1, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, "r-", alpha=0.7,
            label=f"Fit: R={r_value:.3f}")

    ax.set_xlabel("Actual Difficulty (IRT)", fontsize=12)
    ax.set_ylabel("Predicted Difficulty", fontsize=12)
    ax.set_title("Predicted vs Actual Benchmark Difficulty", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="upper left")
    ax.set_aspect("equal")

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(
    importances: dict[str, float],
    output_path: str,
) -> None:
    """Plot Random Forest feature importances.

    Args:
        importances: Dict mapping feature name -> importance score.
        output_path: Path to save the figure.
    """
    sorted_items = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    names = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    ax.barh(range(len(names)), values, color=colors, edgecolor="white")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Feature Importance (MDI)", fontsize=12)
    ax.set_title("Random Forest Feature Importances for Difficulty Prediction",
                 fontsize=14)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
