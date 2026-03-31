"""Tests for analysis and visualization."""

import json
import os
import tempfile
import pytest

from src.analysis import (
    compute_summary_statistics,
    identify_privacy_cliff,
    plot_privacy_utility_curve,
)


def _make_mock_results() -> dict:
    """Create mock results for testing."""
    baseline_runs = [
        {"accuracy": 0.90, "epsilon": float("inf"), "seed": 42},
        {"accuracy": 0.88, "epsilon": float("inf"), "seed": 123},
        {"accuracy": 0.92, "epsilon": float("inf"), "seed": 456},
    ]

    dp_runs = []
    # Low noise -> high accuracy, high epsilon
    for seed in [42, 123, 456]:
        dp_runs.append({
            "accuracy": 0.85, "epsilon": 100.0,
            "noise_multiplier": 0.1, "max_norm": 1.0, "seed": seed,
        })
    # High noise -> low accuracy, low epsilon
    for seed in [42, 123, 456]:
        dp_runs.append({
            "accuracy": 0.25, "epsilon": 0.5,
            "noise_multiplier": 10.0, "max_norm": 1.0, "seed": seed,
        })
    # Medium noise
    for seed in [42, 123, 456]:
        dp_runs.append({
            "accuracy": 0.60, "epsilon": 5.0,
            "noise_multiplier": 1.0, "max_norm": 1.0, "seed": seed,
        })

    return {
        "baseline_runs": baseline_runs,
        "dp_runs": dp_runs,
    }


class TestComputeSummaryStatistics:
    """Tests for summary statistics computation."""

    def test_baseline_stats(self):
        results = _make_mock_results()
        summary = compute_summary_statistics(results)
        assert abs(summary["baseline_accuracy_mean"] - 0.9) < 0.01

    def test_configuration_count(self):
        results = _make_mock_results()
        summary = compute_summary_statistics(results)
        # 3 noise levels x 1 clipping norm = 3 configs
        assert len(summary["configurations"]) == 3

    def test_seeds_aggregated(self):
        results = _make_mock_results()
        summary = compute_summary_statistics(results)
        for cfg in summary["configurations"]:
            assert cfg["n_seeds"] == 3


class TestIdentifyPrivacyCliff:
    """Tests for privacy cliff detection."""

    def test_detects_cliff(self):
        results = _make_mock_results()
        summary = compute_summary_statistics(results)
        cliff = identify_privacy_cliff(summary)
        # Should detect that high-noise configs collapse
        assert cliff["n_configs_below_threshold"] > 0

    def test_cliff_epsilon_is_reasonable(self):
        results = _make_mock_results()
        summary = compute_summary_statistics(results)
        cliff = identify_privacy_cliff(summary)
        if cliff["cliff_epsilon"] is not None:
            assert cliff["cliff_epsilon"] > 0

    def test_no_cliff_when_no_configuration_collapses(self):
        results = _make_mock_results()
        for run in results["dp_runs"]:
            run["accuracy"] = max(run["accuracy"], 0.60)

        summary = compute_summary_statistics(results)
        cliff = identify_privacy_cliff(summary)

        assert cliff["n_configs_below_threshold"] == 0
        assert cliff["cliff_epsilon"] is None
        assert cliff["cliff_accuracy"] is None
        assert cliff["safe_epsilon"] is not None


class TestPlotting:
    """Tests for plot generation."""

    def test_privacy_utility_plot(self):
        results = _make_mock_results()
        summary = compute_summary_statistics(results)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_plot.png")
            plot_privacy_utility_curve(summary, path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
