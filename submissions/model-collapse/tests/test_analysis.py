"""Tests for analysis and reporting."""

import numpy as np
import pytest

from src.analysis import (
    aggregate_by_condition,
    anchor_effectiveness,
    build_summary,
    classify_curve,
)
from src.simulation import SimConfig, _run_single


def _make_results(n_gen: int = 5) -> list[dict]:
    """Run a small set of simulations for testing."""
    configs = [
        SimConfig("naive", 0.0, "bimodal", 42, n_gen),
        SimConfig("naive", 0.0, "bimodal", 123, n_gen),
        SimConfig("anchored", 0.05, "bimodal", 42, n_gen),
        SimConfig("anchored", 0.05, "bimodal", 123, n_gen),
        SimConfig("naive", 0.05, "bimodal", 42, n_gen),
        SimConfig("naive", 0.05, "bimodal", 123, n_gen),
    ]
    return [_run_single(c) for c in configs]


class TestAggregation:
    """aggregate_by_condition correctness."""

    def test_groups_correctly(self) -> None:
        results = _make_results()
        agg = aggregate_by_condition(results)
        assert len(agg) == 3
        for key, val in agg.items():
            assert len(val["mean_kl"]) == 5
            assert len(val["std_kl"]) == 5

    def test_mean_between_runs(self) -> None:
        results = _make_results()
        agg = aggregate_by_condition(results)
        key = ("naive", 0.0, "bimodal")
        r_42 = [r for r in results if r["config"]["seed"] == 42 and r["config"]["agent_type"] == "naive" and r["config"]["gt_fraction"] == 0.0][0]
        r_123 = [r for r in results if r["config"]["seed"] == 123 and r["config"]["agent_type"] == "naive" and r["config"]["gt_fraction"] == 0.0][0]
        kl_42_0 = r_42["generations"][0]["kl_divergence"]
        kl_123_0 = r_123["generations"][0]["kl_divergence"]
        mean_kl = agg[key]["mean_kl"][0]
        assert min(kl_42_0, kl_123_0) <= mean_kl <= max(kl_42_0, kl_123_0)


class TestCurveClassification:
    """classify_curve shape detection."""

    def test_stable_curve(self) -> None:
        kl = np.array([0.05, 0.06, 0.05, 0.07, 0.06])
        result = classify_curve(kl)
        assert result["shape"] == "stable"

    def test_exponential_curve(self) -> None:
        x = np.arange(10, dtype=float)
        kl = 0.1 * np.exp(0.4 * x)
        result = classify_curve(kl)
        assert result["shape"] == "exponential"

    def test_linear_curve(self) -> None:
        kl = np.linspace(0.1, 2.0, 10)
        result = classify_curve(kl)
        assert result["shape"] == "linear"


class TestAnchorEffectiveness:
    """anchor_effectiveness computation."""

    def test_has_all_agents(self) -> None:
        results = _make_results()
        agg = aggregate_by_condition(results)
        eff = anchor_effectiveness(agg)
        assert "naive" in eff
        assert "anchored" in eff

    def test_baseline_has_no_delta(self) -> None:
        results = _make_results()
        agg = aggregate_by_condition(results)
        eff = anchor_effectiveness(agg)
        baselines = [e for e in eff["naive"] if e["gt_fraction"] == 0.0]
        for b in baselines:
            assert b["delta_per_pct"] is None


class TestSummary:
    """build_summary table."""

    def test_row_count(self) -> None:
        results = _make_results()
        agg = aggregate_by_condition(results)
        summary = build_summary(agg)
        assert len(summary) == 3

    def test_row_fields(self) -> None:
        results = _make_results()
        agg = aggregate_by_condition(results)
        summary = build_summary(agg)
        for row in summary:
            assert "agent_type" in row
            assert "gt_fraction" in row
            assert "final_kl_mean" in row
            assert "curve_shape" in row
            assert row["curve_shape"] in ("exponential", "linear", "stable")
