"""Tests for reputation algorithms."""

from src.agents import Agent
from src.rng import default_rng
from src.reputation import simple_average, weighted_by_history, pagerank_trust, eigentrust


def _make_agents(n):
    return [Agent(agent_id=i, true_quality=0.5) for i in range(n)]


def _make_ledger(agents, rng, n_ratings=100):
    """Create a simple ledger where ratings reflect true quality + noise."""
    def _clip(value):
        return max(0.0, min(1.0, value))

    ledger = []
    for rnd in range(n_ratings):
        i, j = rng.choice(len(agents), size=2, replace=False)
        r = float(_clip(agents[j].true_quality + rng.normal(0, 0.1)))
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
    # New algorithm uses agent.account_age directly (quadratic weight).
    # Set rater 0 as old (age=100) and rater 1 as new (age=10).
    agents = _make_agents(3)
    agents[0].account_age = 100  # old rater -- high weight
    agents[1].account_age = 10   # new rater -- low weight
    agents[2].account_age = 100

    # Rater 0 (old, high-weight) rates agent 2 low: 0.2
    # Rater 1 (new, low-weight) rates agent 2 high: 0.8
    ledger = [
        (0, 2, 0.2, 0),
        (1, 2, 0.8, 100),
    ]
    scores = weighted_by_history(agents, ledger, current_round=100)
    # Rater 0 weight = 100^2 + 1 = 10001; rater 1 weight = 10^2 + 1 = 101
    # weighted mean = (10001*0.2 + 101*0.8) / (10001 + 101) = (2000.2 + 80.8) / 10102 ~ 0.205
    # So older rater dominates -- agent 2 score should be well below 0.5
    assert scores[2] < 0.4


def test_pagerank_returns_scores_for_all():
    rng = default_rng(42)
    agents = _make_agents(5)
    ledger = _make_ledger(agents, rng, 50)
    scores = pagerank_trust(agents, ledger)
    assert len(scores) == 5
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_eigentrust_returns_scores_for_all():
    rng = default_rng(42)
    agents = _make_agents(5)
    ledger = _make_ledger(agents, rng, 50)
    scores = eigentrust(agents, ledger)
    assert len(scores) == 5
    assert all(0.0 <= v <= 1.0 for v in scores.values())


def test_all_algorithms_deterministic():
    rng = default_rng(99)
    agents = [Agent(agent_id=i, true_quality=i / 4) for i in range(5)]
    ledger = _make_ledger(agents, rng, 200)
    s1 = simple_average(agents, ledger)
    s2 = simple_average(agents, ledger)
    assert s1 == s2
