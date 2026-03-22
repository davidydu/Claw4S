"""Tests for metrics module."""

import numpy as np
import pytest

from src.metrics import gini_coefficient, compute_round_metrics, compute_summary_metrics


def test_gini_perfect_equality():
    assert gini_coefficient(np.array([1.0, 1.0, 1.0, 1.0])) == pytest.approx(0.0)


def test_gini_maximal_inequality():
    g = gini_coefficient(np.array([0.0, 0.0, 0.0, 1.0]))
    assert g > 0.5  # high inequality


def test_gini_all_zeros():
    assert gini_coefficient(np.array([0.0, 0.0, 0.0])) == 0.0


def test_gini_single_element():
    assert gini_coefficient(np.array([5.0])) == 0.0


def test_compute_round_metrics_keys():
    payoffs = np.array([1.0, 2.0, 3.0, 4.0])
    sharing = np.array([0.5, 0.5, 0.5, 0.5])
    errors = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_round_metrics(payoffs, sharing, errors)
    assert "mean_sharing" in m
    assert "group_welfare" in m
    assert "welfare_gap" in m
    assert "info_asymmetry" in m
    assert "mean_error" in m


def test_compute_round_metrics_values():
    payoffs = np.array([1.0, 2.0, 3.0, 4.0])
    sharing = np.array([0.2, 0.4, 0.6, 0.8])
    errors = np.array([0.1, 0.2, 0.3, 0.4])
    m = compute_round_metrics(payoffs, sharing, errors)
    assert m["mean_sharing"] == pytest.approx(0.5)
    assert m["group_welfare"] == pytest.approx(10.0)
    assert m["welfare_gap"] == pytest.approx(3.0)


def test_compute_summary_metrics():
    n_rounds = 100
    agent_types = ["Open", "Secretive"]
    round_metrics = [
        {"mean_sharing": 0.5, "group_welfare": 2.0, "welfare_gap": 0.5,
         "info_asymmetry": 0.1, "mean_error": 0.2}
        for _ in range(n_rounds)
    ]
    per_agent_sharing = [np.array([1.0, 0.0]) for _ in range(n_rounds)]
    per_agent_payoffs = [np.array([1.5, 0.5]) for _ in range(n_rounds)]

    summary = compute_summary_metrics(
        round_metrics, agent_types, per_agent_sharing, per_agent_payoffs, n_rounds
    )
    assert summary["avg_sharing_rate"] == pytest.approx(0.5)
    assert summary["avg_group_welfare"] == pytest.approx(2.0)
    assert summary["per_type_sharing"]["Open"] == pytest.approx(1.0)
    assert summary["per_type_sharing"]["Secretive"] == pytest.approx(0.0)
