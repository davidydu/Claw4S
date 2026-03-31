"""Generate publication-quality figures for benchmark correlation analysis."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import dendrogram

from src.data import BENCHMARKS, get_model_names, get_model_families, get_model_params


# Consistent color palette for model families
FAMILY_COLORS = {
    "Llama-2": "#e41a1c",
    "Llama-1": "#ff7f00",
    "Mistral": "#984ea3",
    "Falcon": "#377eb8",
    "Pythia": "#4daf4a",
    "OPT": "#a65628",
    "GPT-NeoX": "#f781bf",
    "GPT-Neo": "#999999",
    "Cerebras-GPT": "#66c2a5",
    "MPT": "#fc8d62",
    "StableLM": "#8da0cb",
}


def plot_correlation_heatmap(results, save_path="results/figures/correlation.png"):
    """Plot Pearson and Spearman correlation heatmaps side by side."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    corr_data = results["correlation"]
    benchmarks = corr_data["benchmarks"]
    short_names = [b.replace("-Challenge", "-C") for b in benchmarks]

    for ax, key, title in [
        (axes[0], "pearson", "Pearson Correlation"),
        (axes[1], "spearman", "Spearman Correlation"),
    ]:
        matrix = np.array(corr_data[key])
        im = ax.imshow(matrix, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(short_names)))
        ax.set_yticks(range(len(short_names)))
        ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
        ax.set_yticklabels(short_names, fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold")

        # Annotate cells
        for i in range(len(benchmarks)):
            for j in range(len(benchmarks)):
                val = matrix[i, j]
                color = "white" if abs(val) > 0.7 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=color)

    fig.colorbar(im, ax=axes, shrink=0.8, label="Correlation")
    fig.suptitle("Benchmark Correlation Matrix (N=40 models)", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pca_variance(results, save_path="results/figures/pca_variance.png"):
    """Plot explained variance ratio and cumulative variance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pca = results["pca"]
    evr = pca["explained_variance_ratio"]
    cumvar = pca["cumulative_variance"]
    n_components = len(evr)

    # Bar chart of individual variance
    ax = axes[0]
    bars = ax.bar(range(1, n_components + 1), [v * 100 for v in evr],
                  color="#4daf4a", edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Principal Component", fontsize=11)
    ax.set_ylabel("Variance Explained (%)", fontsize=11)
    ax.set_title("Individual Variance", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, n_components + 1))
    for bar, v in zip(bars, evr):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)

    # Cumulative variance line
    ax = axes[1]
    ax.plot(range(1, n_components + 1), [v * 100 for v in cumvar],
            "o-", color="#e41a1c", markersize=8, linewidth=2)
    ax.axhline(y=90, color="gray", linestyle="--", alpha=0.7, label="90% threshold")
    ax.axhline(y=95, color="gray", linestyle=":", alpha=0.7, label="95% threshold")
    n90 = pca["n_components_90"]
    n95 = pca["n_components_95"]
    ax.axvline(x=n90, color="#377eb8", linestyle="--", alpha=0.5)
    ax.annotate(f"{n90} PCs for 90%", xy=(n90, 90), xytext=(n90 + 0.3, 85),
                fontsize=9, color="#377eb8")
    ax.set_xlabel("Number of Components", fontsize=11)
    ax.set_ylabel("Cumulative Variance (%)", fontsize=11)
    ax.set_title("Cumulative Variance Explained", fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, n_components + 1))
    ax.set_ylim(50, 102)
    ax.legend(fontsize=9)

    fig.suptitle("PCA of Benchmark Scores", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_pca(results, save_path="results/figures/model_pca.png"):
    """Plot models in PC1-PC2 space, colored by family, sized by params."""
    fig, ax = plt.subplots(figsize=(12, 8))

    pca = results["pca"]
    model_pcs = np.array(pca["model_pcs"])
    names = get_model_names()
    families = get_model_families()
    params = get_model_params()

    # Size: log-scaled parameters
    sizes = 30 + 80 * (np.log10(params) - np.log10(params.min())) / (
        np.log10(params.max()) - np.log10(params.min())
    )

    for fam in sorted(set(families)):
        mask = [i for i, f in enumerate(families) if f == fam]
        color = FAMILY_COLORS.get(fam, "#333333")
        ax.scatter(
            model_pcs[mask, 0], model_pcs[mask, 1],
            s=sizes[mask], c=color, label=fam, alpha=0.8,
            edgecolors="black", linewidth=0.5, zorder=3,
        )

    # Label selected models (largest/smallest per family, plus notable ones)
    label_models = {
        "Llama-2-70B", "Llama-2-7B", "Mistral-7B", "Falcon-40B",
        "Pythia-12B", "Pythia-70M", "OPT-66B", "OPT-125M",
        "GPT-NeoX-20B", "Cerebras-GPT-13B", "Cerebras-GPT-111M",
        "MPT-30B",
    }
    for i, name in enumerate(names):
        if name in label_models:
            ax.annotate(
                name, (model_pcs[i, 0], model_pcs[i, 1]),
                xytext=(5, 5), textcoords="offset points",
                fontsize=7, alpha=0.8,
            )

    evr = pca["explained_variance_ratio"]
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Models in Principal Component Space", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_dendrogram(results, save_path="results/figures/dendrogram.png"):
    """Plot hierarchical clustering dendrogram of benchmarks."""
    fig, ax = plt.subplots(figsize=(10, 6))

    clust = results["clustering"]
    Z = np.array(clust["linkage"])
    benchmarks = clust["benchmarks"]
    short_names = [b.replace("-Challenge", "-C") for b in benchmarks]

    dendrogram(
        Z, labels=short_names, ax=ax,
        leaf_rotation=0, leaf_font_size=11,
        color_threshold=0.7 * max(Z[:, 2]),
    )
    ax.set_ylabel("Correlation Distance (1 - |r|)", fontsize=11)
    ax.set_title("Hierarchical Clustering of Benchmarks", fontsize=13, fontweight="bold")
    ax.axhline(y=0.7 * max(Z[:, 2]), color="gray", linestyle="--", alpha=0.5,
               label="Cluster threshold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_redundancy(results, save_path="results/figures/redundancy.png"):
    """Plot greedy selection variance explained curve."""
    fig, ax = plt.subplots(figsize=(10, 6))

    red = results["redundancy"]
    order = red["greedy_selection_order"]
    varexp = red["greedy_variance_explained"]

    bars = ax.bar(range(1, len(order) + 1), [v * 100 for v in varexp],
                  color="#377eb8", edgecolor="black", linewidth=0.5)
    ax.axhline(y=90, color="red", linestyle="--", alpha=0.7, label="90% threshold")
    ax.axhline(y=95, color="orange", linestyle="--", alpha=0.7, label="95% threshold")

    ax.set_xticks(range(1, len(order) + 1))
    short_order = [b.replace("-Challenge", "-C") for b in order]
    ax.set_xticklabels(
        [f"{i+1}. {n}" for i, n in enumerate(short_order)],
        rotation=30, ha="right", fontsize=9,
    )
    ax.set_ylabel("Total Variance Explained (%)", fontsize=11)
    ax.set_xlabel("Benchmarks Added (greedy order)", fontsize=11)
    ax.set_title("Greedy Benchmark Selection: Variance Explained", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 105)

    for bar, v in zip(bars, varexp):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f"{v*100:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def generate_all_plots(results):
    """Generate all 5 figures."""
    plot_correlation_heatmap(results)
    plot_pca_variance(results)
    plot_model_pca(results)
    plot_dendrogram(results)
    plot_redundancy(results)
    print("[plots] Generated 5 figures in results/figures/")
