"""Tests for agent creation."""

from src.agents import create_honest_agents, create_sybil_agents
from src.rng import default_rng


def test_honest_agents_count():
    rng = default_rng(42)
    agents = create_honest_agents(10, rng)
    assert len(agents) == 10


def test_honest_agents_quality_range():
    rng = default_rng(42)
    agents = create_honest_agents(20, rng)
    for a in agents:
        assert 0.2 <= a.true_quality <= 0.9
        assert not a.is_sybil


def test_honest_agents_unique_ids():
    rng = default_rng(42)
    agents = create_honest_agents(15, rng)
    ids = [a.agent_id for a in agents]
    assert len(set(ids)) == 15


def test_sybil_agents_count():
    sybils = create_sybil_agents(5, start_id=20, controller_id=999)
    assert len(sybils) == 5


def test_sybil_agents_properties():
    sybils = create_sybil_agents(3, start_id=100, controller_id=42)
    for s in sybils:
        assert s.is_sybil
        assert s.sybil_controller == 42
        assert s.true_quality == 0.1
    assert [s.agent_id for s in sybils] == [100, 101, 102]
