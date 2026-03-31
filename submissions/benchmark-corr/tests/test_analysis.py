"""Tests for src/analysis.py — verify analysis functions produce valid results."""

import math
import numpy as np
from src.data import BENCHMARKS, SCORES
from src.analysis import (
    compute_correlation_matrices,
    run_pca,
    run_clustering,
    analyze_redundancy,
    analyze_model_families,
    run_full_analysis,
)
from src.report import generate_report


def test_correlation_matrices_symmetric():
    """Correlation matrices should be symmetric."""
    result = compute_correlation_matrices(SCORES, seed=42)
    for key in ["pearson", "spearman"]:
        m = np.array(result[key])
        np.testing.assert_allclose(m, m.T, atol=1e-10)


def test_correlation_diagonal_is_one():
    """Diagonal of correlation matrices should be 1.0."""
    result = compute_correlation_matrices(SCORES, seed=42)
    for key in ["pearson", "spearman"]:
        m = np.array(result[key])
        np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-10)


def test_correlation_values_in_range():
    """All correlations should be in [-1, 1]."""
    result = compute_correlation_matrices(SCORES, seed=42)
    for key in ["pearson", "spearman"]:
        m = np.array(result[key])
        assert np.all(m >= -1.0 - 1e-10)
        assert np.all(m <= 1.0 + 1e-10)


def test_pca_variance_sums_to_one():
    """PCA explained variance ratios should sum to ~1.0."""
    result = run_pca(SCORES, seed=42)
    total = sum(result["explained_variance_ratio"])
    assert abs(total - 1.0) < 1e-6


def test_pca_n_components_90():
    """PCA should need <= 4 components for 90% variance (redundancy thesis)."""
    result = run_pca(SCORES, seed=42)
    assert result["n_components_90"] <= 4


def test_pca_cumulative_monotonic():
    """Cumulative variance should be monotonically increasing."""
    result = run_pca(SCORES, seed=42)
    cumvar = result["cumulative_variance"]
    for i in range(1, len(cumvar)):
        assert cumvar[i] >= cumvar[i - 1]


def test_clustering_valid_clusters():
    """Clustering should assign valid cluster labels."""
    result = run_clustering(SCORES, seed=42)
    assert len(result["clusters_2"]) == len(BENCHMARKS)
    assert len(result["clusters_3"]) == len(BENCHMARKS)
    assert set(result["clusters_2"]).issubset({1, 2})
    assert set(result["clusters_3"]).issubset({1, 2, 3})


def test_clustering_distance_matrix():
    """Distance matrix should be symmetric, non-negative, zero diagonal."""
    result = run_clustering(SCORES, seed=42)
    dist = np.array(result["distance_matrix"])
    np.testing.assert_allclose(dist, dist.T, atol=1e-10)
    assert np.all(dist >= -1e-10)
    np.testing.assert_allclose(np.diag(dist), 0.0, atol=1e-10)


def test_clustering_records_average_linkage_method():
    """Clustering should record the valid linkage used for correlation distances."""
    result = run_clustering(SCORES, seed=42)
    assert result["linkage_method"] == "average"
    assert result["distance_metric"] == "1 - abs(correlation)"


def test_redundancy_greedy_order():
    """Greedy selection should return all benchmarks in some order."""
    result = analyze_redundancy(SCORES, seed=42)
    order = result["greedy_selection_order"]
    assert set(order) == set(BENCHMARKS)
    assert len(order) == len(BENCHMARKS)


def test_redundancy_variance_monotonic():
    """Greedy variance explained should be monotonically increasing."""
    result = analyze_redundancy(SCORES, seed=42)
    varexp = result["greedy_variance_explained"]
    for i in range(1, len(varexp)):
        assert varexp[i] >= varexp[i - 1] - 1e-10


def test_family_analysis_silhouette():
    """Silhouette score should be in [-1, 1]."""
    result = analyze_model_families(SCORES, seed=42)
    sil = result["silhouette_score"]
    assert math.isfinite(sil)
    assert -1.0 <= sil <= 1.0


def test_family_analysis_pc1_correlation():
    """PC1 should correlate significantly with log(params)."""
    result = analyze_model_families(SCORES, seed=42)
    assert math.isfinite(result["pc1_param_correlation"])
    assert abs(result["pc1_param_correlation"]) > 0.5


def test_full_analysis_returns_all_sections():
    """run_full_analysis should return all expected sections."""
    result = run_full_analysis(seed=42)
    expected_keys = {"metadata", "correlation", "pca", "clustering",
                     "redundancy", "family_analysis", "robustness"}
    assert set(result.keys()) == expected_keys


def test_full_analysis_deterministic():
    """Running with same seed should produce identical results."""
    r1 = run_full_analysis(seed=42)
    r2 = run_full_analysis(seed=42)
    # Compare PCA variance ratios
    assert r1["pca"]["explained_variance_ratio"] == r2["pca"]["explained_variance_ratio"]
    # Compare correlation matrices
    np.testing.assert_allclose(
        np.array(r1["correlation"]["pearson"]),
        np.array(r2["correlation"]["pearson"]),
        atol=1e-10,
    )


def test_report_mentions_only_analyzed_benchmarks():
    """Generated report should not reference benchmarks outside the submission."""
    report = generate_report(run_full_analysis(seed=42))
    assert "PIQA" not in report


def test_full_analysis_metadata_has_reproducibility_fingerprint():
    """Metadata should include a deterministic data fingerprint for reproducibility."""
    result = run_full_analysis(seed=42)
    fp = result["metadata"]["data_fingerprint_sha256"]
    assert len(fp) == 64
    assert all(ch in "0123456789abcdef" for ch in fp)


def test_robustness_section_has_expected_structure():
    """Bootstrap robustness summary should include CI and selection stability fields."""
    result = run_full_analysis(seed=42)
    robustness = result["robustness"]
    assert robustness["n_bootstrap_samples"] >= 100
    assert set(robustness["n_components_90_distribution"].keys())
    assert "top2_selection_frequencies" in robustness
    assert len(robustness["top2_selection_frequencies"]) >= 1
    assert "pc1_param_correlation_ci95" in robustness
    ci = robustness["pc1_param_correlation_ci95"]
    assert len(ci) == 2
    assert ci[0] <= ci[1]
