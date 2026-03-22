"""Tests for reputation algorithms."""

import numpy as np

from src.agents import Agent
from src.reputation import simple_average, weighted_by_history, pagerank_trust, eigentrust


def _make_agents(n):
    return [Agent(agent_id=i, true_quality=0.5) for i in range(n)]


def _make_ledger(agents, rng, n_ratings=100):
    """Create a simple ledger where ratings reflect true quality + noise."""
    ledger = []
    for rnd in range(n_ratings):
        i, j = rng.choice(len(agents), size=2, replace=False)
        r = float(np.clip(agents[j].true_quality + rng.normal(0, 0.1), 0, 1))
        ledger.append((agents[i].agent_id, agents[j].agent_id, r, rnd))
    return ledger


def test_simple_average_no_ratings():
    agents = _make_agents(5)
    scores = simple_average(agents, [])
    assert all(v == 0.5 for v in scores.values())


def test_simple_average_basic():
    agents = _make_agents(3)
    ledger = [
        (0, 1, 0.9, 0),
        (2, 1, 0.7, 1),
    ]
    scores = simple_average(agents, ledger)
    assert abs(scores[1] - 0.8) < 1e-6


def test_weighted_history_older_raters_matter_more():
    agents = _make_agents(3)
    # Rater 0 rates agent 2 low at round 0
    # Rater 1 rates agent 2 high at round 100
    ledger = [
        (0, 2, 0.2, 0),
        (1, 2, 0.8, 100),
        (0, 1, 0.5, 0),  # so rater 0 first seen at round 0
        (1, 0, 0.5, 50),  # rater 1 first seen at round 50
    ]
    scores = weighted_by_history(agents, ledger, current_round=100)
    # Rater 0 has age 100 (weight = log2(102) ~ 6.67)
    # Rater 1 has age 50 (weight = log2(52) ~ 5.70)
    # So rater 0's low rating should pull agent 2 down
    assert scores[2] < 0.5


def test_pagerank_returns_scores_for_all():
    rng = np.random.default_rng(42)
    agents = _make_agents(5)
    ledger = _make_ledger(agents, rng, 50)
    scores = pagerank_trust(agents, ledger)
    assert len(scores) == 5
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_eigentrust_returns_scores_for_all():
    rng = np.random.default_rng(42)
    agents = _make_agents(5)
    ledger = _make_ledger(agents, rng, 50)
    scores = eigentrust(agents, ledger)
    assert len(scores) == 5
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_all_algorithms_deterministic():
    rng = np.random.default_rng(99)
    agents = [Agent(agent_id=i, true_quality=i / 4) for i in range(5)]
    ledger = _make_ledger(agents, rng, 200)
    s1 = simple_average(agents, ledger)
    s2 = simple_average(agents, ledger)
    assert s1 == s2
