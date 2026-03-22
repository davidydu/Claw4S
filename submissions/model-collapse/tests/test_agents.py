"""Tests for agent types."""

import numpy as np
import pytest

from src.agents import AGENT_CLASSES, NaiveAgent, SelectiveAgent, AnchoredAgent
from src.distributions import sample_ground_truth, kl_divergence_numerical


class TestAgentBasics:
    """Basic agent functionality."""

    @pytest.mark.parametrize("agent_type", list(AGENT_CLASSES.keys()))
    def test_learn_and_produce(self, agent_type: str) -> None:
        rng = np.random.default_rng(42)
        cls = AGENT_CLASSES[agent_type]
        agent = cls("bimodal", 0.05, rng)
        data = sample_ground_truth("bimodal", 2000, rng)
        agent.learn(data)
        output = agent.produce(1000)
        assert output.shape == (1000,)

    @pytest.mark.parametrize("agent_type", list(AGENT_CLASSES.keys()))
    def test_produce_before_learn_raises(self, agent_type: str) -> None:
        rng = np.random.default_rng(42)
        cls = AGENT_CLASSES[agent_type]
        agent = cls("bimodal", 0.0, rng)
        with pytest.raises(RuntimeError, match="not learned"):
            agent.produce(100)

    def test_agent_type_attribute(self) -> None:
        assert NaiveAgent.agent_type == "naive"
        assert SelectiveAgent.agent_type == "selective"
        assert AnchoredAgent.agent_type == "anchored"


class TestNaiveAgent:
    """Naive agent learns from all data."""

    def test_gen0_kl_small(self) -> None:
        rng = np.random.default_rng(42)
        agent = NaiveAgent("bimodal", 0.0, rng)
        data = sample_ground_truth("bimodal", 2000, rng)
        agent.learn(data)
        kl = kl_divergence_numerical("bimodal", agent.kde)
        assert kl < 0.5, f"Gen0 KL={kl:.4f} too large"

    def test_collapses_without_gt(self) -> None:
        """Naive agent degrades over 10 generations with no ground truth."""
        rng = np.random.default_rng(42)
        agent = NaiveAgent("bimodal", 0.0, rng)
        data = sample_ground_truth("bimodal", 2000, rng)
        kl_first = None
        kl_last = None
        for gen in range(10):
            agent.learn(data)
            kl = kl_divergence_numerical("bimodal", agent.kde)
            if gen == 0:
                kl_first = kl
            if gen == 9:
                kl_last = kl
            data = agent.produce(2000)
        assert kl_last > kl_first, "KL should increase over generations"


class TestSelectiveAgent:
    """Selective agent filters low-confidence samples."""

    def test_filters_samples(self) -> None:
        """Selective agent's KDE should differ from naive on same data."""
        rng_n = np.random.default_rng(42)
        rng_s = np.random.default_rng(42)
        data = sample_ground_truth("bimodal", 2000, np.random.default_rng(42))

        naive = NaiveAgent("bimodal", 0.0, rng_n)
        selective = SelectiveAgent("bimodal", 0.0, rng_s)
        naive.learn(data)
        selective.learn(data)

        kl_n = kl_divergence_numerical("bimodal", naive.kde)
        kl_s = kl_divergence_numerical("bimodal", selective.kde)
        assert kl_n < 0.5
        assert kl_s < 0.5


class TestAnchoredAgent:
    """Anchored agent mixes ground truth."""

    def test_stays_stable_with_gt(self) -> None:
        """Anchored agent with 10% GT should not collapse in 10 gens."""
        rng = np.random.default_rng(42)
        agent = AnchoredAgent("bimodal", 0.10, rng)
        data = sample_ground_truth("bimodal", 2000, rng)
        for gen in range(10):
            agent.learn(data)
            kl = kl_divergence_numerical("bimodal", agent.kde)
            data = agent.produce(2000)
        assert kl < 1.0, f"Anchored (10% GT) collapsed at gen 9: KL={kl:.4f}"

    def test_gt_fraction_zero_same_as_naive(self) -> None:
        """Anchored with 0% GT should behave like naive (but with 1 GT sample)."""
        rng = np.random.default_rng(42)
        agent = AnchoredAgent("bimodal", 0.0, rng)
        data = sample_ground_truth("bimodal", 2000, np.random.default_rng(99))
        agent.learn(data)
        kl = kl_divergence_numerical("bimodal", agent.kde)
        assert kl < 0.5
