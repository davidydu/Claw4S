# src/plots.py
"""Generate publication-quality plots for Zipf analysis.

Uses matplotlib with Agg backend (no display required).
"""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_zipf_fit(
    ranks: np.ndarray,
    freqs: np.ndarray,
    fit_params: dict,
    title: str,
    output_path: str,
) -> None:
    """Plot log-log rank-frequency data with fitted Zipf-Mandelbrot line.

    Args:
        ranks: Rank values (1-indexed).
        freqs: Corresponding frequencies.
        fit_params: Dict with alpha, q, C, r_squared.
        title: Plot title.
        output_path: Path to save PNG.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot empirical data
    ax.loglog(ranks, freqs, "o", markersize=3, alpha=0.5, label="Empirical data")

    # Plot fitted line
    alpha = fit_params["alpha"]
    q = fit_params["q"]
    C = fit_params["C"]
    r_squared = fit_params["r_squared"]

    if C > 0:
        fitted = C / (ranks.astype(float) + q) ** alpha
        ax.loglog(
            ranks,
            fitted,
            "-",
            color="red",
            linewidth=2,
            label=f"Zipf-Mandelbrot fit\n"
            f"  alpha={alpha:.3f}, q={q:.1f}\n"
            f"  R^2={r_squared:.4f}",
        )

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_piecewise_comparison(
    results: list[dict],
    output_path: str,
) -> None:
    """Plot grouped bar chart comparing head/body/tail Zipf exponents.

    Args:
        results: List of dicts with 'label' and 'piecewise_fit' keys.
        output_path: Path to save PNG.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    labels = [r["label"] for r in results]
    regions = ["head", "body", "tail"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    n = len(labels)
    x = np.arange(n)
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(10, n * 1.5), 6))

    for i, (region, color) in enumerate(zip(regions, colors)):
        alphas = [r["piecewise_fit"][region]["alpha"] for r in results]
        ax.bar(x + i * width, alphas, width, label=f"{region.capitalize()}", color=color)

    ax.set_xlabel("Corpus x Tokenizer", fontsize=12)
    ax.set_ylabel("Zipf Exponent (alpha)", fontsize=12)
    ax.set_title("Piecewise Zipf Exponents by Region", fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=10)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Zipf (alpha=1)")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_alpha_compression_correlation(
    alphas: list[float],
    compressions: list[float],
    labels: list[str],
    output_path: str,
) -> None:
    """Plot scatter of Zipf exponent vs compression ratio with trend line.

    Args:
        alphas: Zipf exponent values.
        compressions: Compression ratio values (chars/token).
        labels: Labels for each point.
        output_path: Path to save PNG.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.scatter(alphas, compressions, s=60, alpha=0.7, zorder=5)

    # Add labels to points
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (alphas[i], compressions[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=7,
            alpha=0.8,
        )

    # Trend line
    if len(alphas) >= 3:
        z = np.polyfit(alphas, compressions, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(alphas), max(alphas), 50)
        ax.plot(x_line, p(x_line), "--", color="red", alpha=0.5, label=f"Trend: slope={z[0]:.2f}")
        ax.legend(fontsize=10)

    ax.set_xlabel("Zipf Exponent (alpha)", fontsize=12)
    ax.set_ylabel("Compression Ratio (chars/token)", fontsize=12)
    ax.set_title("Zipf Exponent vs Tokenizer Compression", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_zipf_overlay(
    all_data: list[dict],
    output_path: str,
) -> None:
    """Overlay multiple rank-frequency distributions on one log-log plot.

    Args:
        all_data: List of dicts with 'label', 'ranks', 'freqs' keys.
        output_path: Path to save PNG.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 7))

    for data in all_data:
        ax.loglog(
            data["ranks"],
            data["freqs"],
            "o-",
            markersize=2,
            alpha=0.6,
            label=data["label"],
        )

    ax.set_xlabel("Rank", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Token Rank-Frequency Distributions (Log-Log)", fontsize=14)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
