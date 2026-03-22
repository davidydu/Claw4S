"""Tests for src/agents.py — agent types."""

import numpy as np
import pytest

from src.agents import (
    StubbornAgent, AdaptiveAgent, LeaderAgent, FollowerAgent,
    make_agent, AGENT_TYPES,
)


@pytest.fixture
def peaked_prior():
    """A clearly peaked prior: action 2 is strongly preferred."""
    p = np.array([0.05, 0.05, 0.75, 0.05, 0.10])
    return p / p.sum()


class TestStubbornAgent:
    def test_always_plays_peak(self, peaked_prior):
        agent = StubbornAgent(agent_id=0, prior=peaked_prior, seed=42)
        for _ in range(20):
            assert agent.choose_action() == 2

    def test_beliefs_never_change(self, peaked_prior):
        agent = StubbornAgent(agent_id=0, prior=peaked_prior, seed=42)
        agent.choose_action()
        agent.update([0, 0, 0, 0])  # everyone plays 0
        np.testing.assert_array_equal(agent.beliefs, peaked_prior)


class TestAdaptiveAgent:
    def test_greedy_action_is_peak(self, peaked_prior):
        # With epsilon=0 (no exploration), should always play peak
        agent = AdaptiveAgent(agent_id=0, prior=peaked_prior, seed=42, epsilon=0.0)
        assert agent.choose_action() == 2

    def test_beliefs_update_toward_observed(self, peaked_prior):
        agent = AdaptiveAgent(agent_id=0, prior=peaked_prior, seed=42,
                              learning_rate=0.5, epsilon=0.0)
        agent.choose_action()
        # Everyone plays action 0
        agent.update([0, 0, 0, 0])
        # After update, belief for action 0 should increase
        assert agent.beliefs[0] > peaked_prior[0]

    def test_converges_to_group_action(self, peaked_prior):
        agent = AdaptiveAgent(agent_id=0, prior=peaked_prior, seed=42,
                              learning_rate=0.3, epsilon=0.0)
        for _ in range(100):
            agent.choose_action()
            agent.update([0, 0, 0, 0])
        # After 100 rounds of seeing everyone play 0, should prefer 0
        assert agent.preferred_action() == 0

    def test_beliefs_stay_normalised(self, peaked_prior):
        agent = AdaptiveAgent(agent_id=0, prior=peaked_prior, seed=42)
        for _ in range(50):
            agent.choose_action()
            agent.update([1, 2, 3, 4])
        np.testing.assert_allclose(agent.beliefs.sum(), 1.0, atol=1e-10)

    def test_epsilon_greedy_explores(self, peaked_prior):
        agent = AdaptiveAgent(agent_id=0, prior=peaked_prior, seed=42, epsilon=1.0)
        actions = [agent.choose_action() for _ in range(100)]
        # With epsilon=1.0, should explore all actions
        assert len(set(actions)) > 1


class TestLeaderAgent:
    def test_always_plays_peak(self, peaked_prior):
        agent = LeaderAgent(agent_id=0, prior=peaked_prior, seed=42)
        for _ in range(20):
            assert agent.choose_action() == 2

    def test_beliefs_never_change(self, peaked_prior):
        agent = LeaderAgent(agent_id=0, prior=peaked_prior, seed=42)
        agent.choose_action()
        agent.update([0, 0, 0, 0])
        np.testing.assert_array_equal(agent.beliefs, peaked_prior)


class TestFollowerAgent:
    def test_high_learning_rate(self, peaked_prior):
        agent = FollowerAgent(agent_id=0, prior=peaked_prior, seed=42)
        assert agent.learning_rate == 0.5  # default is high

    def test_converges_faster_than_adaptive(self, peaked_prior):
        follower = FollowerAgent(agent_id=0, prior=peaked_prior.copy(), seed=42)
        adaptive = AdaptiveAgent(agent_id=1, prior=peaked_prior.copy(), seed=42)

        for _ in range(10):
            follower.choose_action()
            adaptive.choose_action()
            follower.update([0, 0, 0, 0])
            adaptive.update([0, 0, 0, 0])

        # Follower should have higher belief for action 0
        assert follower.beliefs[0] > adaptive.beliefs[0]


class TestFactory:
    def test_make_all_types(self, peaked_prior):
        for name in AGENT_TYPES:
            agent = make_agent(name, agent_id=0, prior=peaked_prior, seed=42)
            assert agent.agent_type == AGENT_TYPES[name].__name__

    def test_invalid_type_raises(self, peaked_prior):
        with pytest.raises(KeyError):
            make_agent("nonexistent", agent_id=0, prior=peaked_prior)

    def test_history_recorded(self, peaked_prior):
        agent = make_agent("adaptive", agent_id=0, prior=peaked_prior, seed=42)
        for _ in range(5):
            agent.choose_action()
            agent.update([0, 0, 0, 0])
        assert len(agent.history) == 5
