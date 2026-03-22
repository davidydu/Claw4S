"""Tests for agent types."""

import numpy as np
import pytest

from src.agents import (
    OpenAgent,
    SecretiveAgent,
    ReciprocalAgent,
    StrategicAgent,
    create_agents,
    AGENT_TYPES,
)


def test_open_always_shares_full():
    rng = np.random.default_rng(42)
    a = OpenAgent(0, rng)
    for t in range(100):
        assert a.choose_disclosure(t, [0.0], [0.0]) == 1.0


def test_secretive_never_shares():
    rng = np.random.default_rng(42)
    a = SecretiveAgent(0, rng)
    for t in range(100):
        assert a.choose_disclosure(t, [0.0], [1.0]) == 0.0


def test_reciprocal_tracks_others():
    rng = np.random.default_rng(42)
    a = ReciprocalAgent(0, rng, alpha=0.5)
    # Others share a lot -> reciprocal should increase
    for _ in range(20):
        a.choose_disclosure(0, [0.0], [0.9])
    assert a.choose_disclosure(0, [0.0], [0.9]) > 0.7


def test_reciprocal_decreases_when_others_hoard():
    rng = np.random.default_rng(42)
    a = ReciprocalAgent(0, rng, alpha=0.5)
    for _ in range(20):
        a.choose_disclosure(0, [0.0], [0.1])
    assert a.choose_disclosure(0, [0.0], [0.1]) < 0.3


def test_strategic_output_in_range():
    rng = np.random.default_rng(42)
    a = StrategicAgent(0, rng)
    for t in range(50):
        dl = a.choose_disclosure(t, [0.0] * t, [0.5] * t)
        assert 0.0 <= dl <= 1.0


def test_create_agents_factory():
    rng = np.random.default_rng(42)
    agents = create_agents(["open", "secretive", "reciprocal", "strategic"], rng)
    assert len(agents) == 4
    assert agents[0].type_name() == "Open"
    assert agents[1].type_name() == "Secretive"
    assert agents[2].type_name() == "Reciprocal"
    assert agents[3].type_name() == "Strategic"


def test_all_types_registered():
    assert set(AGENT_TYPES.keys()) == {"open", "secretive", "reciprocal", "strategic"}
