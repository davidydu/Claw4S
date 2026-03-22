"""Core analysis: correlation, PCA, clustering, redundancy."""

import numpy as np
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.data import BENCHMARKS, SCORES, get_model_families, get_model_params


def compute_correlation_matrices(scores, seed=42):
    """Compute Pearson and Spearman correlation matrices between benchmarks.

    Args:
        scores: (n_models, n_benchmarks) array of scores.
        seed: Random seed (unused here, kept for API consistency).

    Returns:
        dict with 'pearson' and 'spearman' correlation matrices,
        plus 'pearson_pvalues' and 'spearman_pvalues'.
    """
    n_benchmarks = scores.shape[1]
    pearson_corr = np.zeros((n_benchmarks, n_benchmarks))
    pearson_pval = np.zeros((n_benchmarks, n_benchmarks))
    spearman_corr = np.zeros((n_benchmarks, n_benchmarks))
    spearman_pval = np.zeros((n_benchmarks, n_benchmarks))

    for i in range(n_benchmarks):
        for j in range(n_benchmarks):
            r, p = stats.pearsonr(scores[:, i], scores[:, j])
            pearson_corr[i, j] = r
            pearson_pval[i, j] = p
            rho, p_s = stats.spearmanr(scores[:, i], scores[:, j])
            spearman_corr[i, j] = rho
            spearman_pval[i, j] = p_s

    return {
        "pearson": pearson_corr,
        "pearson_pvalues": pearson_pval,
        "spearman": spearman_corr,
        "spearman_pvalues": spearman_pval,
        "benchmarks": list(BENCHMARKS),
    }


def run_pca(scores, seed=42):
    """Run PCA on standardized benchmark scores.

    Args:
        scores: (n_models, n_benchmarks) array.
        seed: Random seed for reproducibility.

    Returns:
        dict with explained_variance_ratio, cumulative_variance,
        components, n_components_90, n_components_95.
    """
    np.random.seed(seed)
    scaler = StandardScaler()
    scores_scaled = scaler.fit_transform(scores)

    pca = PCA(random_state=seed)
    pca.fit(scores_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n90 = int(np.searchsorted(cumvar, 0.90) + 1)
    n95 = int(np.searchsorted(cumvar, 0.95) + 1)

    # Project models into PC space for clustering
    model_pcs = pca.transform(scores_scaled)

    return {
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "cumulative_variance": cumvar.tolist(),
        "components": pca.components_.tolist(),
        "n_components_90": n90,
        "n_components_95": n95,
        "model_pcs": model_pcs.tolist(),
        "loadings": pca.components_.tolist(),
        "benchmarks": list(BENCHMARKS),
    }


def run_clustering(scores, seed=42):
    """Hierarchical clustering of benchmarks based on correlation distance.

    Args:
        scores: (n_models, n_benchmarks) array.
        seed: Random seed (unused, deterministic).

    Returns:
        dict with linkage matrix, cluster assignments, dendrogram order.
    """
    # Correlation-based distance: d = 1 - |r|
    corr = np.corrcoef(scores.T)
    dist = 1.0 - np.abs(corr)
    # Ensure symmetry and zero diagonal
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0
    # Clamp small negative values from floating point
    dist = np.clip(dist, 0.0, None)

    condensed = squareform(dist)
    Z = linkage(condensed, method="ward")

    # Cut at 2 and 3 clusters for analysis
    clusters_2 = fcluster(Z, t=2, criterion="maxclust").tolist()
    clusters_3 = fcluster(Z, t=3, criterion="maxclust").tolist()

    return {
        "linkage": Z.tolist(),
        "clusters_2": clusters_2,
        "clusters_3": clusters_3,
        "distance_matrix": dist.tolist(),
        "benchmarks": list(BENCHMARKS),
    }


def analyze_redundancy(scores, seed=42):
    """Determine which benchmarks are most/least redundant.

    For each benchmark, compute average absolute correlation with all others.
    Identify the minimal set that captures 90%+ variance via greedy selection.

    Args:
        scores: (n_models, n_benchmarks) array.
        seed: Random seed.

    Returns:
        dict with redundancy rankings, greedy selection order,
        and conditional variance explained.
    """
    np.random.seed(seed)
    corr = np.corrcoef(scores.T)
    n = len(BENCHMARKS)

    # Average absolute correlation with OTHER benchmarks
    avg_abs_corr = []
    for i in range(n):
        others = [abs(corr[i, j]) for j in range(n) if j != i]
        avg_abs_corr.append(float(np.mean(others)))

    # Greedy forward selection: pick benchmark that explains most variance
    scaler = StandardScaler()
    scores_scaled = scaler.fit_transform(scores)

    remaining = list(range(n))
    selected = []
    variance_explained = []

    for step in range(n):
        best_var = -1.0
        best_idx = -1
        for idx in remaining:
            candidate = selected + [idx]
            sub = scores_scaled[:, candidate]
            pca = PCA(random_state=seed)
            pca.fit(sub)
            # Total variance of full data explained by these benchmarks
            # Use projection approach
            reconstructed = pca.inverse_transform(pca.transform(sub))
            # But we want variance of ALL benchmarks explained
            # Use correlation-based approach instead
            from numpy.linalg import lstsq
            # Regress all benchmarks on selected subset
            X = scores_scaled[:, candidate]
            Y = scores_scaled
            coeffs, _, _, _ = lstsq(X, Y, rcond=None)
            Y_pred = X @ coeffs
            ss_res = np.sum((Y - Y_pred) ** 2)
            ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
            r2 = 1.0 - ss_res / ss_tot
            if r2 > best_var:
                best_var = r2
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)
        variance_explained.append(float(best_var))

    greedy_order = [BENCHMARKS[i] for i in selected]

    # Rank by redundancy (most redundant = highest avg correlation with others)
    redundancy_ranking = sorted(
        zip(BENCHMARKS, avg_abs_corr),
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "avg_abs_correlation": {BENCHMARKS[i]: avg_abs_corr[i] for i in range(n)},
        "redundancy_ranking": [(name, float(val)) for name, val in redundancy_ranking],
        "greedy_selection_order": greedy_order,
        "greedy_variance_explained": variance_explained,
        "benchmarks": list(BENCHMARKS),
    }


def analyze_model_families(scores, seed=42):
    """Analyze whether models from the same family cluster together in PC space.

    Args:
        scores: (n_models, n_benchmarks) array.
        seed: Random seed.

    Returns:
        dict with silhouette score, family centroids in PC space,
        intra-family vs inter-family distances.
    """
    np.random.seed(seed)
    families = get_model_families()
    params = get_model_params()

    scaler = StandardScaler()
    scores_scaled = scaler.fit_transform(scores)

    pca = PCA(n_components=2, random_state=seed)
    model_pcs = pca.fit_transform(scores_scaled)

    # Get unique families with >= 2 members for meaningful analysis
    unique_families = sorted(set(families))
    family_counts = {f: families.count(f) for f in unique_families}
    multi_member = [f for f in unique_families if family_counts[f] >= 2]

    # Compute centroids for families with >= 2 members
    centroids = {}
    for fam in multi_member:
        indices = [i for i, f in enumerate(families) if f == fam]
        centroids[fam] = model_pcs[indices].mean(axis=0).tolist()

    # Intra-family vs inter-family average distance
    intra_dists = []
    for fam in multi_member:
        indices = [i for i, f in enumerate(families) if f == fam]
        if len(indices) < 2:
            continue
        for ii in range(len(indices)):
            for jj in range(ii + 1, len(indices)):
                d = np.linalg.norm(model_pcs[indices[ii]] - model_pcs[indices[jj]])
                intra_dists.append(float(d))

    inter_dists = []
    for i in range(len(multi_member)):
        for j in range(i + 1, len(multi_member)):
            c1 = np.array(centroids[multi_member[i]])
            c2 = np.array(centroids[multi_member[j]])
            inter_dists.append(float(np.linalg.norm(c1 - c2)))

    # Silhouette-like metric (only for multi-member families)
    from sklearn.metrics import silhouette_score as sklearn_silhouette
    multi_mask = [i for i, f in enumerate(families) if f in multi_member]
    if len(multi_mask) >= 4 and len(set(families[i] for i in multi_mask)) >= 2:
        labels = [families[i] for i in multi_mask]
        sil = float(sklearn_silhouette(model_pcs[multi_mask], labels))
    else:
        sil = float("nan")

    # Parameter-performance correlation (log params vs PC1)
    log_params = np.log10(params)
    pc1_corr, pc1_pval = stats.pearsonr(log_params, model_pcs[:, 0])

    return {
        "silhouette_score": sil,
        "family_centroids": centroids,
        "avg_intra_family_distance": float(np.mean(intra_dists)) if intra_dists else float("nan"),
        "avg_inter_family_distance": float(np.mean(inter_dists)) if inter_dists else float("nan"),
        "n_multi_member_families": len(multi_member),
        "multi_member_families": multi_member,
        "pc1_param_correlation": float(pc1_corr),
        "pc1_param_pvalue": float(pc1_pval),
        "pca_variance_explained": pca.explained_variance_ratio_.tolist(),
    }


def run_full_analysis(seed=42):
    """Run all analyses and return combined results dict.

    Args:
        seed: Random seed for all stochastic components.

    Returns:
        dict with all analysis results, suitable for JSON serialization.
    """
    np.random.seed(seed)
    scores = SCORES.copy()

    correlation = compute_correlation_matrices(scores, seed=seed)
    pca_results = run_pca(scores, seed=seed)
    clustering = run_clustering(scores, seed=seed)
    redundancy = analyze_redundancy(scores, seed=seed)
    family_analysis = analyze_model_families(scores, seed=seed)

    return {
        "metadata": {
            "n_models": int(scores.shape[0]),
            "n_benchmarks": int(scores.shape[1]),
            "benchmarks": list(BENCHMARKS),
            "seed": seed,
        },
        "correlation": correlation,
        "pca": pca_results,
        "clustering": clustering,
        "redundancy": redundancy,
        "family_analysis": family_analysis,
    }
