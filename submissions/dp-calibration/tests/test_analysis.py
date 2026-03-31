"""Unit tests for the analysis module."""

from importlib import metadata

import pytest
from src.analysis import run_analysis, T_VALUES, DELTA_VALUES, SIGMA_VALUES


class TestRunAnalysis:
    """Tests for the full analysis pipeline."""

    @pytest.fixture(scope="class")
    def analysis_data(self):
        """Run analysis once for all tests in this class."""
        return run_analysis(seed=42)

    def test_metadata_completeness(self, analysis_data):
        """Metadata should contain all expected fields."""
        meta = analysis_data["metadata"]
        assert meta["num_T"] == len(T_VALUES)
        assert meta["num_delta"] == len(DELTA_VALUES)
        assert meta["num_sigma"] == len(SIGMA_VALUES)
        assert meta["num_methods"] == 4
        expected = len(T_VALUES) * len(DELTA_VALUES) * len(SIGMA_VALUES)
        assert meta["total_configs"] == expected
        assert meta["elapsed_seconds"] >= 0

    def test_metadata_includes_reproducibility_manifest(self, analysis_data):
        """Metadata should include deterministic digest + package versions."""
        meta = analysis_data["metadata"]
        assert "results_digest" in meta
        assert len(meta["results_digest"]) == 64
        assert "package_versions" in meta
        versions = meta["package_versions"]
        assert versions["numpy"] == metadata.version("numpy")
        assert versions["scipy"] == metadata.version("scipy")
        assert versions["matplotlib"] == metadata.version("matplotlib")

    def test_result_count(self, analysis_data):
        """Should have one result per grid point."""
        expected = len(T_VALUES) * len(DELTA_VALUES) * len(SIGMA_VALUES)
        assert len(analysis_data["results"]) == expected

    def test_result_structure(self, analysis_data):
        """Each result should have required fields."""
        for r in analysis_data["results"]:
            assert "T" in r
            assert "delta" in r
            assert "sigma" in r
            assert "epsilons" in r
            assert "best_epsilon" in r
            assert "best_method" in r
            assert "tightness_ratio" in r

    def test_tightness_ratios_valid(self, analysis_data):
        """All tightness ratios should be >= 1.0."""
        for r in analysis_data["results"]:
            for method, ratio in r["tightness_ratio"].items():
                assert ratio >= 0.999, (
                    f"Ratio {ratio} < 1.0 for {method} at "
                    f"T={r['T']}, sigma={r['sigma']}")

    def test_summary_win_counts(self, analysis_data):
        """Win counts should sum to total configs."""
        summary = analysis_data["summary"]
        total_wins = sum(summary["win_counts"].values())
        total_configs = analysis_data["metadata"]["total_configs"]
        assert total_wins == total_configs

    def test_summary_includes_robust_tightness_stats(self, analysis_data):
        """Summary should include median/p95 tightness metrics per method."""
        summary = analysis_data["summary"]
        for method in ("naive", "advanced", "rdp", "gdp"):
            median = summary["median_tightness_ratio"][method]
            p95 = summary["p95_tightness_ratio"][method]
            assert median >= 0.999
            assert p95 >= median

    def test_reproducible(self):
        """Running with same seed should give identical results."""
        data1 = run_analysis(seed=42)
        data2 = run_analysis(seed=42)
        for r1, r2 in zip(data1["results"], data2["results"]):
            assert r1["epsilons"] == r2["epsilons"]
        assert data1["metadata"]["results_digest"] == data2["metadata"]["results_digest"]

    def test_custom_grid_override(self):
        """run_analysis should support user-specified grids."""
        data = run_analysis(
            seed=7,
            t_values=[5, 50],
            delta_values=[1e-5],
            sigma_values=[0.5, 1.0],
        )
        meta = data["metadata"]
        assert meta["num_T"] == 2
        assert meta["num_delta"] == 1
        assert meta["num_sigma"] == 2
        assert meta["total_configs"] == 4
        assert data["grid"]["T_values"] == [5, 50]
        assert data["grid"]["delta_values"] == [1e-5]
        assert data["grid"]["sigma_values"] == [0.5, 1.0]

    def test_fast_runtime(self, analysis_data):
        """Analysis should complete in under 30 seconds."""
        assert analysis_data["metadata"]["elapsed_seconds"] < 30.0
