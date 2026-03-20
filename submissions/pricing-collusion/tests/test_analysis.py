# tests/test_analysis.py
import numpy as np
from src.analysis import compute_statistics


def _make_records(n_seeds=3, avg_price=1.8, nash=1.5, monopoly=2.0):
    """Create mock analysis records for testing."""
    records = []
    for seed in range(n_seeds):
        records.append({
            "matchup": "QQ", "memory": 1, "preset": "e-commerce",
            "shocks": False, "seed": seed,
            "final_avg_price": avg_price + (seed - 1) * 0.05,
            "nash_price": nash, "monopoly_price": monopoly,
            "convergence_round": 100_000,
            "pre_shock_price": None, "post_shock_price": None,
            "recovery_rounds": None,
            "auditor_scores": {"margin": 0.8, "deviation_punishment": 0.6,
                               "counterfactual": 0.7, "welfare": 0.75},
            "panel_majority": True, "panel_unanimous": False,
            "panel_weighted": True,
        })
    return records


def test_compute_statistics_groups_by_condition():
    """Should produce one stat entry per unique condition."""
    records = _make_records()
    stats = compute_statistics(records)
    assert len(stats) == 1
    assert stats[0]["matchup"] == "QQ"
    assert stats[0]["n_seeds"] == 3


def test_compute_statistics_t_test_above_nash():
    """With prices above Nash, p-value should be significant."""
    records = _make_records(avg_price=1.8, nash=1.5)
    stats = compute_statistics(records)
    assert stats[0]["p_value"] < 0.05
    assert stats[0]["cohens_d"] > 0


def test_compute_statistics_t_test_at_nash():
    """With prices at Nash, should not be significant."""
    records = _make_records(avg_price=1.5, nash=1.5)
    stats = compute_statistics(records)
    # Prices right at Nash — not significantly above
    assert stats[0]["cohens_d"] <= 0.1


def test_compute_statistics_collusion_rates():
    """Majority and unanimous rates should be between 0 and 1."""
    records = _make_records()
    stats = compute_statistics(records)
    assert 0.0 <= stats[0]["majority_collusion_rate"] <= 1.0
    assert 0.0 <= stats[0]["unanimous_collusion_rate"] <= 1.0
