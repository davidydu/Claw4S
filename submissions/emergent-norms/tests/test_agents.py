"""Tests for agent behaviors."""

import numpy as np

from src.agents import Agent, AgentType, create_population
from src.game import NUM_ACTIONS


def test_conformist_follows_majority():
    """Conformist picks the action with highest population count."""
    rng = np.random.default_rng(42)
    agent = Agent(AgentType.CONFORMIST, 0, rng)
    # Population overwhelmingly plays action 2
    pop_counts = np.array([1, 0, 50])
    actions = [agent.choose_action(pop_counts, rng) for _ in range(20)]
    assert all(a == 2 for a in actions)


def test_traditionalist_anchors():
    """Traditionalist locks onto first action giving payoff >= threshold."""
    rng = np.random.default_rng(42)
    agent = Agent(AgentType.TRADITIONALIST, 0, rng, anchor_threshold=2.0)
    assert agent.anchor_action is None

    # Give a low payoff — should not anchor
    agent.update(1, 1.0)
    assert agent.anchor_action is None

    # Give a high payoff — should anchor on action 0
    agent.update(0, 3.0)
    assert agent.anchor_action == 0

    # Subsequent choices should all be action 0
    pop_counts = np.array([5, 5, 5])
    actions = [agent.choose_action(pop_counts, rng) for _ in range(20)]
    assert all(a == 0 for a in actions)


def test_innovator_explores():
    """Innovator explores at least sometimes over many rounds."""
    rng = np.random.default_rng(42)
    agent = Agent(AgentType.INNOVATOR, 0, rng, epsilon=0.5)
    pop_counts = np.array([10, 10, 10])
    actions = set()
    for _ in range(100):
        a = agent.choose_action(pop_counts, rng)
        actions.add(a)
    # With epsilon=0.5 over 100 rounds, should see multiple actions
    assert len(actions) >= 2


def test_adaptive_learns():
    """Adaptive agent shifts beliefs toward high-payoff action."""
    rng = np.random.default_rng(42)
    agent = Agent(AgentType.ADAPTIVE, 0, rng, ema_alpha=0.3, temperature=0.5)

    # Repeatedly reward action 1
    for _ in range(50):
        agent.update(1, 5.0)

    # Beliefs should be concentrated on action 1
    assert agent.beliefs[1] > agent.beliefs[0]
    assert agent.beliefs[1] > agent.beliefs[2]


def test_create_population_size():
    """create_population returns correct number of agents."""
    rng = np.random.default_rng(42)
    comp = {AgentType.ADAPTIVE: 10, AgentType.CONFORMIST: 5}
    agents = create_population(comp, rng)
    assert len(agents) == 15


def test_create_population_types():
    """create_population assigns correct agent types."""
    rng = np.random.default_rng(42)
    comp = {AgentType.ADAPTIVE: 3, AgentType.INNOVATOR: 2}
    agents = create_population(comp, rng)
    types = [a.agent_type for a in agents]
    assert types.count(AgentType.ADAPTIVE) == 3
    assert types.count(AgentType.INNOVATOR) == 2


def test_agent_action_in_range():
    """All agent types produce actions in [0, NUM_ACTIONS)."""
    rng = np.random.default_rng(42)
    pop_counts = np.array([5, 5, 5])
    for atype in AgentType:
        agent = Agent(atype, 0, rng)
        for _ in range(50):
            a = agent.choose_action(pop_counts, rng)
            assert 0 <= a < NUM_ACTIONS


def test_agent_avg_payoff():
    """avg_payoff returns correct running mean."""
    rng = np.random.default_rng(42)
    agent = Agent(AgentType.ADAPTIVE, 0, rng)
    agent.update(0, 2.0)
    agent.update(0, 4.0)
    assert agent.avg_payoff == 3.0
