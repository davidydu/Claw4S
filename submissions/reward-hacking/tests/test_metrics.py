"""Tests for metrics computation."""

import numpy as np
import pytest

from src.metrics import compute_summary_metrics, aggregate_across_seeds


class TestComputeSummaryMetrics:
    def _make_result(self, adoption_curve, divergence_curve=None):
        n = len(adoption_curve)
        if divergence_curve is None:
            divergence_curve = [0.1] * n
        return {
            "adoption_curve": adoption_curve,
            "proxy_reward_curve": [1.0] * n,
            "true_reward_curve": [0.9] * n,
            "divergence_curve": divergence_curve,
            "containment_events": 5,
            "final_adoption": adoption_curve[-1] if adoption_curve else 0.0,
            "time_to_50pct": None,
            "time_to_90pct": None,
        }

    def test_steady_state_adoption(self):
        # Last 20% is all 0.8
        curve = [0.0] * 80 + [0.8] * 20
        result = self._make_result(curve)
        m = compute_summary_metrics(result)
        assert abs(m["steady_state_adoption"] - 0.8) < 0.01

    def test_containment_success_true(self):
        curve = [0.0] * 100
        result = self._make_result(curve)
        m = compute_summary_metrics(result)
        assert m["containment_success"] is True

    def test_containment_success_false(self):
        curve = [1.0] * 100
        result = self._make_result(curve)
        m = compute_summary_metrics(result)
        assert m["containment_success"] is False

    def test_peak_adoption(self):
        curve = [0.0, 0.5, 1.0, 0.7, 0.3]
        result = self._make_result(curve)
        m = compute_summary_metrics(result)
        assert m["peak_adoption"] == 1.0

    def test_values_bounded(self):
        curve = [0.3] * 100
        result = self._make_result(curve)
        m = compute_summary_metrics(result)
        assert 0.0 <= m["steady_state_adoption"] <= 1.0
        assert 0.0 <= m["peak_adoption"] <= 1.0


class TestAggregateAcrossSeeds:
    def test_mean_and_std(self):
        results = [
            {"steady_state_adoption": 0.8, "peak_adoption": 1.0,
             "steady_state_divergence": 0.5, "final_adoption": 0.8,
             "containment_success": True, "containment_events": 3,
             "time_to_50pct": 100, "time_to_90pct": 200},
            {"steady_state_adoption": 0.6, "peak_adoption": 0.9,
             "steady_state_divergence": 0.3, "final_adoption": 0.6,
             "containment_success": False, "containment_events": 1,
             "time_to_50pct": 150, "time_to_90pct": None},
        ]
        agg = aggregate_across_seeds(results)
        assert abs(agg["steady_state_adoption_mean"] - 0.7) < 0.01
        assert agg["containment_rate"] == 0.5
        assert agg["time_to_50pct_reached_frac"] == 1.0
        assert agg["time_to_90pct_reached_frac"] == 0.5

    def test_empty(self):
        assert aggregate_across_seeds([]) == {}
