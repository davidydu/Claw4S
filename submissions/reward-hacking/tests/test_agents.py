"""Tests for agent creation and reward sampling."""

import numpy as np
import pytest

from src.agents import (
    Agent,
    create_agent_population,
    sample_reward,
    HACK_PROXY_MEAN,
    HONEST_PROXY_MEAN,
)


class TestCreatePopulation:
    def test_correct_count(self):
        agents = create_agent_population(10, 0.0, np.random.default_rng(42))
        assert len(agents) == 10

    def test_monitor_fraction(self):
        agents = create_agent_population(10, 0.5, np.random.default_rng(42))
        n_monitors = sum(1 for a in agents if a.agent_type == "monitor")
        assert n_monitors == 5

    def test_zero_monitors(self):
        agents = create_agent_population(10, 0.0, np.random.default_rng(42))
        n_monitors = sum(1 for a in agents if a.agent_type == "monitor")
        assert n_monitors == 0

    def test_all_agents_start_honest(self):
        agents = create_agent_population(10, 0.25, np.random.default_rng(42))
        for a in agents:
            assert a.strategy == "honest"

    def test_non_monitors_have_three_types(self):
        agents = create_agent_population(12, 0.25, np.random.default_rng(42))
        types = {a.agent_type for a in agents if a.agent_type != "monitor"}
        assert types == {"explorer", "imitator", "conservative"}

    def test_unique_ids(self):
        agents = create_agent_population(10, 0.1, np.random.default_rng(42))
        ids = [a.agent_id for a in agents]
        assert len(set(ids)) == 10

    def test_deterministic(self):
        a1 = create_agent_population(10, 0.25, np.random.default_rng(42))
        a2 = create_agent_population(10, 0.25, np.random.default_rng(42))
        for x, y in zip(a1, a2):
            assert x.agent_type == y.agent_type


class TestSampleReward:
    def test_honest_reward_range(self):
        rng = np.random.default_rng(42)
        agent = Agent(agent_id=0, agent_type="explorer", strategy="honest")
        rewards = [sample_reward(agent, rng) for _ in range(100)]
        proxy_mean = np.mean([r[0] for r in rewards])
        assert abs(proxy_mean - HONEST_PROXY_MEAN) < 0.1

    def test_hack_reward_higher_proxy(self):
        rng = np.random.default_rng(42)
        honest = Agent(agent_id=0, agent_type="explorer", strategy="honest")
        hacker = Agent(agent_id=1, agent_type="explorer", strategy="hack")
        h_rewards = [sample_reward(honest, rng)[0] for _ in range(200)]
        k_rewards = [sample_reward(hacker, rng)[0] for _ in range(200)]
        assert np.mean(k_rewards) > np.mean(h_rewards)

    def test_hack_reward_lower_true(self):
        rng = np.random.default_rng(42)
        honest = Agent(agent_id=0, agent_type="explorer", strategy="honest")
        hacker = Agent(agent_id=1, agent_type="explorer", strategy="hack")
        h_true = [sample_reward(honest, rng)[1] for _ in range(200)]
        k_true = [sample_reward(hacker, rng)[1] for _ in range(200)]
        assert np.mean(k_true) < np.mean(h_true)


class TestAgentProperties:
    def test_beta_explorer(self):
        a = Agent(agent_id=0, agent_type="explorer")
        assert a.beta == 0.5

    def test_beta_monitor_zero(self):
        a = Agent(agent_id=0, agent_type="monitor")
        assert a.beta == 0.0

    def test_is_hacking(self):
        a = Agent(agent_id=0, agent_type="explorer", strategy="hack")
        assert a.is_hacking is True

    def test_is_not_hacking(self):
        a = Agent(agent_id=0, agent_type="explorer", strategy="honest")
        assert a.is_hacking is False
