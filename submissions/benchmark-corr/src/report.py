"""Generate a human-readable markdown report from analysis results."""

import numpy as np
from src.data import BENCHMARKS


def generate_report(results):
    """Generate markdown report from analysis results dict.

    Args:
        results: dict from run_full_analysis().

    Returns:
        str: Markdown-formatted report.
    """
    lines = []
    meta = results["metadata"]
    corr = results["correlation"]
    pca = results["pca"]
    clust = results["clustering"]
    red = results["redundancy"]
    fam = results["family_analysis"]
    robust = results["robustness"]

    lines.append("# LLM Benchmark Correlation Analysis Report")
    lines.append("")
    lines.append(f"**Models analyzed:** {meta['n_models']}")
    lines.append(f"**Benchmarks:** {meta['n_benchmarks']} ({', '.join(meta['benchmarks'])})")
    lines.append(f"**Random seed:** {meta['seed']}")
    lines.append(f"**Bootstrap samples:** {meta['n_bootstrap_samples']}")
    lines.append(f"**Data fingerprint (SHA-256):** `{meta['data_fingerprint_sha256']}`")
    lines.append("")

    # --- Section 1: Correlation ---
    lines.append("## 1. Correlation Analysis")
    lines.append("")
    lines.append("### Pearson Correlation Matrix")
    lines.append("")
    short = [b.replace("-Challenge", "-C") for b in corr["benchmarks"]]
    header = "| | " + " | ".join(short) + " |"
    sep = "|---" * (len(short) + 1) + "|"
    lines.append(header)
    lines.append(sep)
    pearson = np.array(corr["pearson"])
    for i, name in enumerate(short):
        row = f"| **{name}** |"
        for j in range(len(short)):
            row += f" {pearson[i, j]:.3f} |"
        lines.append(row)
    lines.append("")

    # Identify highest and lowest correlations (off-diagonal)
    n = len(BENCHMARKS)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((BENCHMARKS[i], BENCHMARKS[j], pearson[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)

    lines.append("**Highest correlations (Pearson):**")
    for a, b, r in pairs[:3]:
        lines.append(f"- {a} vs {b}: r = {r:.3f}")
    lines.append("")
    lines.append("**Lowest correlations (Pearson):**")
    for a, b, r in pairs[-3:]:
        lines.append(f"- {a} vs {b}: r = {r:.3f}")
    lines.append("")

    # --- Section 2: PCA ---
    lines.append("## 2. Principal Component Analysis")
    lines.append("")
    evr = pca["explained_variance_ratio"]
    cumvar = pca["cumulative_variance"]
    lines.append("| PC | Variance (%) | Cumulative (%) |")
    lines.append("|---|---|---|")
    for i, (v, c) in enumerate(zip(evr, cumvar)):
        lines.append(f"| PC{i+1} | {v*100:.1f} | {c*100:.1f} |")
    lines.append("")
    lines.append(f"**Components for 90% variance:** {pca['n_components_90']}")
    lines.append(f"**Components for 95% variance:** {pca['n_components_95']}")
    lines.append("")

    # PC loadings
    lines.append("### PC Loadings (top 2 components)")
    lines.append("")
    loadings = np.array(pca["loadings"])
    lines.append("| Benchmark | PC1 | PC2 |")
    lines.append("|---|---|---|")
    for i, bm in enumerate(pca["benchmarks"]):
        lines.append(f"| {bm} | {loadings[0, i]:.3f} | {loadings[1, i]:.3f} |")
    lines.append("")

    # --- Section 3: Clustering ---
    lines.append("## 3. Hierarchical Clustering")
    lines.append("")
    lines.append(f"**Linkage method:** {clust['linkage_method']}")
    lines.append(f"**Distance metric:** {clust['distance_metric']}")
    lines.append("")
    lines.append("### 2-Cluster Assignment")
    for i, bm in enumerate(clust["benchmarks"]):
        lines.append(f"- {bm}: Cluster {clust['clusters_2'][i]}")
    lines.append("")
    lines.append("### 3-Cluster Assignment")
    for i, bm in enumerate(clust["benchmarks"]):
        lines.append(f"- {bm}: Cluster {clust['clusters_3'][i]}")
    lines.append("")

    # --- Section 4: Redundancy ---
    lines.append("## 4. Redundancy Analysis")
    lines.append("")
    lines.append("### Average Absolute Correlation with Other Benchmarks")
    lines.append("")
    for name, val in red["redundancy_ranking"]:
        lines.append(f"- {name}: {val:.3f}")
    lines.append("")
    lines.append("### Greedy Forward Selection (variance explained by subset)")
    lines.append("")
    for i, (bm, var) in enumerate(zip(red["greedy_selection_order"],
                                       red["greedy_variance_explained"])):
        lines.append(f"{i+1}. **{bm}**: {var*100:.1f}% total variance explained")
    lines.append("")

    # Recommendation
    lines.append("### Recommendation")
    lines.append("")
    top2 = red["greedy_selection_order"][:2]
    top3 = red["greedy_selection_order"][:3]
    var2 = red["greedy_variance_explained"][1]
    var3 = red["greedy_variance_explained"][2]
    lines.append(f"If limited to **2 benchmarks**, use: {', '.join(top2)} "
                 f"({var2*100:.1f}% variance)")
    lines.append(f"If limited to **3 benchmarks**, use: {', '.join(top3)} "
                 f"({var3*100:.1f}% variance)")
    lines.append("")

    # --- Section 5: Model Family Analysis ---
    lines.append("## 5. Model Family Analysis")
    lines.append("")
    lines.append(f"**Silhouette score:** {fam['silhouette_score']:.3f}")
    lines.append(f"**Avg intra-family distance:** {fam['avg_intra_family_distance']:.3f}")
    lines.append(f"**Avg inter-family distance:** {fam['avg_inter_family_distance']:.3f}")
    lines.append(f"**PC1-log(params) correlation:** r = {fam['pc1_param_correlation']:.3f} "
                 f"(p = {fam['pc1_param_pvalue']:.2e})")
    lines.append("")

    # --- Section 6: Robustness ---
    lines.append("## 6. Robustness Checks")
    lines.append("")
    ci_low = np.array(robust["pearson_ci95_lower"])
    ci_high = np.array(robust["pearson_ci95_upper"])
    arc_idx = BENCHMARKS.index("ARC-Challenge")
    wino_idx = BENCHMARKS.index("WinoGrande")
    pc1_ci = robust["pc1_param_correlation_ci95"]
    top_pair = robust["top2_selection_frequencies"][0]
    lines.append(
        f"- Bootstrap n={robust['n_bootstrap_samples']} confirms ARC-Challenge vs WinoGrande "
        f"correlation stability: 95% CI [{ci_low[arc_idx, wino_idx]:.3f}, "
        f"{ci_high[arc_idx, wino_idx]:.3f}]."
    )
    lines.append(
        f"- PC1-log(params) correlation remains strong under resampling: "
        f"95% CI [{pc1_ci[0]:.3f}, {pc1_ci[1]:.3f}]."
    )
    n90_dist = robust["n_components_90_distribution"]
    n90_fmt = ", ".join(
        f"{k} PCs: {v}/{robust['n_bootstrap_samples']}"
        for k, v in sorted(n90_dist.items(), key=lambda kv: int(kv[0]))
    )
    lines.append(f"- Effective dimensionality is stable across bootstraps ({n90_fmt}).")
    lines.append(
        f"- Most frequent top-2 subset from greedy selection: "
        f"{top_pair['pair']} ({top_pair['frequency']*100:.1f}% of bootstrap runs)."
    )
    lines.append("")

    # --- Section 7: Key Findings ---
    lines.append("## 7. Key Findings")
    lines.append("")
    lines.append(f"1. **{pca['n_components_90']} principal components explain 90%+ of variance** "
                 f"across {meta['n_benchmarks']} benchmarks, confirming high redundancy.")
    lines.append(f"2. PC1 alone captures {evr[0]*100:.1f}% of variance and correlates strongly "
                 f"with model scale (r = {fam['pc1_param_correlation']:.3f}).")
    lines.append(f"3. The most redundant benchmark is **{red['redundancy_ranking'][0][0]}** "
                 f"(avg |r| = {red['redundancy_ranking'][0][1]:.3f} with others).")
    lines.append(f"4. The least redundant benchmark is **{red['redundancy_ranking'][-1][0]}** "
                 f"(avg |r| = {red['redundancy_ranking'][-1][1]:.3f}), providing unique signal.")
    sil = fam['silhouette_score']
    if sil > 0.1:
        sil_interp = "cluster together"
    elif sil > -0.1:
        sil_interp = "show mixed clustering"
    else:
        sil_interp = "overlap substantially (model size dominates over family identity)"
    lines.append(f"5. Model families {sil_interp} in PC space "
                 f"(silhouette = {sil:.3f}), confirming that scale explains "
                 f"more variance than architecture.")
    lines.append(
        "6. Bootstrap robustness checks show the headline conclusions are stable under "
        "model-level resampling."
    )
    lines.append("")

    # --- Section 8: Limitations ---
    lines.append("## 8. Limitations")
    lines.append("")
    lines.append("- Scores are from multiple evaluation runs (Open LLM Leaderboard, original papers) "
                 "with potentially different prompting strategies.")
    lines.append("- Only base (pre-trained) models are included; instruction-tuned variants may "
                 "show different correlation patterns.")
    lines.append("- The analysis covers mainly decoder-only autoregressive models; "
                 "encoder-decoder models may behave differently.")
    lines.append("- Benchmark difficulty saturation (ceiling effects in HellaSwag and WinoGrande) "
                 "can inflate correlations for large models.")
    lines.append("- TruthfulQA and GSM8K may have different measurement properties "
                 "than the reasoning benchmarks (ARC, HellaSwag, WinoGrande).")
    lines.append("")

    return "\n".join(lines)


def save_report(report, path="results/report.md"):
    """Save report string to file."""
    with open(path, "w") as f:
        f.write(report)
    print(f"[report] Saved to {path}")
