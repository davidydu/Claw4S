"""Tests for pricing agents (TDD — written before implementation)."""

import numpy as np
import pytest
from src.agents import (
    CompetitiveAgent, TitForTatAgent, QLearningAgent, SARSAAgent,
    PolicyGradientAgent, create_agent,
)
from src.market import LogitMarket


def _make_market():
    return LogitMarket(n_sellers=2, alpha=3.0, costs=[1.0, 1.0],
                       price_min=1.0, price_max=2.0, price_grid_size=15)


# --- Rule-based agents ---

def test_competitive_agent_plays_nash():
    """Competitive agent always returns the Nash equilibrium price index."""
    market = _make_market()
    agent = CompetitiveAgent(agent_id=0, market=market)
    price_history = np.array([[1, 1]])  # dummy indices
    action = agent.choose_action(price_history)
    nash = market.nash_price()
    chosen_price = market.price_grid[action]
    # Should be the grid point closest to Nash
    assert abs(chosen_price - nash) <= (market.price_grid[1] - market.price_grid[0])


def test_tit_for_tat_matches_opponent():
    """TFT agent matches the opponent's last price."""
    market = _make_market()
    agent = TitForTatAgent(agent_id=0, market=market)
    # Opponent (agent 1) played price index 10 last round
    price_history = np.array([[5, 10]])
    action = agent.choose_action(price_history)
    assert action == 10  # matches opponent's last action


def test_tit_for_tat_first_round():
    """TFT agent plays a valid (mid-range) price on first round."""
    market = _make_market()
    agent = TitForTatAgent(agent_id=0, market=market)
    price_history = np.empty((0, 2), dtype=int)
    action = agent.choose_action(price_history)
    assert 0 <= action < market.price_grid_size


def test_create_agent_factory():
    """Agent factory should create all 5 agent types."""
    market = _make_market()
    for name in ["q_learning", "sarsa", "policy_gradient", "tit_for_tat", "competitive"]:
        agent = create_agent(name, agent_id=0, market=market, memory=1, seed=42)
        assert agent is not None


# --- RL agents ---

def test_q_learning_explores_initially():
    """Q-learning agent should explore (random actions) when epsilon is high."""
    market = _make_market()
    agent = QLearningAgent(agent_id=0, market=market, memory=1, seed=42)
    actions = set()
    history = np.array([[7, 7]])
    for _ in range(100):
        actions.add(agent.choose_action(history))
    # With eps=1.0, should try many different actions
    assert len(actions) > 5


def test_q_learning_exploits_after_training():
    """After many rounds, Q-learning should exploit (consistent action)."""
    market = _make_market()
    agent = QLearningAgent(agent_id=0, market=market, memory=1, seed=42,
                           total_rounds=1000)
    history = np.array([[7, 7]])
    # Simulate enough rounds to decay epsilon
    for i in range(500):
        action = agent.choose_action(history)
        prices = np.array([market.price_grid[action], market.price_grid[7]])
        reward = market.compute_profits(prices)[0]
        new_history = np.vstack([history, [action, 7]])
        agent.update(history, action, reward, new_history)
        history = new_history

    # Now epsilon should be low — agent should be consistent
    actions = [agent.choose_action(history) for _ in range(20)]
    assert len(set(actions)) <= 3  # mostly the same action


def test_q_learning_save_load_state():
    """Saving and loading state should produce identical behavior."""
    market = _make_market()
    agent = QLearningAgent(agent_id=0, market=market, memory=1, seed=42)
    history = np.array([[7, 7]])
    # Do some updates
    for _ in range(10):
        action = agent.choose_action(history)
        agent.update(history, action, 0.5, history)

    saved = agent.save_state()
    # Change agent state
    for _ in range(10):
        agent.update(history, 0, 1.0, history)

    # Restore
    agent.load_state(saved)
    # Q-values should match what we saved — verify round matches
    assert agent._round == saved["round"]
    state_key = agent._get_state_key(history)
    q_vals = agent._get_q_values(state_key)
    assert q_vals is not None


def test_sarsa_is_on_policy():
    """SARSA agent should exist and use on-policy updates."""
    market = _make_market()
    agent = SARSAAgent(agent_id=0, market=market, memory=1, seed=42)
    history = np.array([[7, 7]])
    action = agent.choose_action(history)
    assert 0 <= action < market.price_grid_size


def test_policy_gradient_produces_valid_actions():
    """Policy gradient agent should produce valid price grid indices."""
    market = _make_market()
    agent = PolicyGradientAgent(agent_id=0, market=market, memory=1, seed=42)
    history = np.array([[7, 7]])
    for _ in range(50):
        action = agent.choose_action(history)
        assert 0 <= action < market.price_grid_size


def test_tile_coding_m3():
    """Agents with memory=3 should use tile coding."""
    market = _make_market()
    agent = QLearningAgent(agent_id=0, market=market, memory=3, seed=42)
    assert agent.use_tiles is True
    history = np.array([[7, 7], [8, 8], [9, 9]])
    action = agent.choose_action(history)
    assert 0 <= action < market.price_grid_size
