"""Tests for Sybil attack strategies."""

from src.agents import Agent, create_honest_agents, create_sybil_agents
from src.rng import default_rng
from src.sybil_strategies import ballot_stuffing, bad_mouthing, whitewashing


def _setup():
    rng = default_rng(42)
    honest = create_honest_agents(10, rng)
    sybils = create_sybil_agents(3, start_id=10, controller_id=999)
    return honest, sybils, rng


def test_ballot_stuffing_returns_ratings():
    honest, sybils, rng = _setup()
    ratings = ballot_stuffing(sybils, honest, rng)
    assert len(ratings) > 0
    # 3 Sybils rate 2 others each = 6 ratings
    assert len(ratings) == 6


def test_ballot_stuffing_high_ratings():
    honest, sybils, rng = _setup()
    ratings = ballot_stuffing(sybils, honest, rng)
    for _, _, value in ratings:
        assert value >= 0.95


def test_bad_mouthing_targets_top_agents():
    honest, sybils, rng = _setup()
    ratings = bad_mouthing(sybils, honest, rng)
    assert len(ratings) > 0
    # Check that some ratings target honest agents with low values
    honest_targeted = [r for r in ratings if r[1] < 10]
    assert any(r[2] < 0.15 for r in honest_targeted)


def test_bad_mouthing_also_inflates_self():
    honest, sybils, rng = _setup()
    ratings = bad_mouthing(sybils, honest, rng)
    sybil_ids = {s.agent_id for s in sybils}
    mutual = [r for r in ratings if r[0] in sybil_ids and r[1] in sybil_ids]
    assert all(r[2] >= 0.9 for r in mutual)


def test_whitewashing_mixed_ratings():
    honest, sybils, rng = _setup()
    ratings = whitewashing(sybils, honest, rng)
    assert len(ratings) > 0
    # Should have both honest-ish and inflated ratings
    values = [r[2] for r in ratings]
    assert min(values) < 0.75
    assert max(values) > 0.8
