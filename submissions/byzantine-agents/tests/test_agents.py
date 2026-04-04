"""Tests for agent implementations."""

import numpy as np
import pytest

from src.agents import (
    K_OPTIONS,
    MajorityVoter,
    BayesianVoter,
    CautiousVoter,
    RandomByzantine,
    StrategicByzantine,
    MimickingByzantine,
    make_honest_agent,
    make_byzantine_agent,
    HONEST_TYPES,
    BYZANTINE_TYPES,
)


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def clear_counts():
    """Count vector where option 2 was observed 3 times, others 0."""
    return np.array([0.0, 0.0, 3.0, 0.0, 0.0])


@pytest.fixture
def ambiguous_counts():
    """Count vector where two options are tied."""
    return np.array([1.0, 1.0, 0.0, 0.0, 0.0])


@pytest.fixture
def zero_counts():
    """Count vector with no observations (all zeros)."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0])


class TestMajorityVoter:
    def test_votes_for_highest_count(self, rng, clear_counts):
        agent = MajorityVoter()
        assert agent.vote(clear_counts, rng) == 2

    def test_deterministic_given_clear_signal(self, rng, clear_counts):
        agent = MajorityVoter()
        votes = [agent.vote(clear_counts, rng) for _ in range(10)]
        assert all(v == 2 for v in votes)

    def test_tie_breaking(self, rng, ambiguous_counts):
        agent = MajorityVoter()
        votes = set(agent.vote(ambiguous_counts, np.random.default_rng(s)) for s in range(100))
        # Should break ties between option 0 and 1
        assert votes.issubset({0, 1})


class TestBayesianVoter:
    def test_votes_for_highest_posterior(self, rng, clear_counts):
        agent = BayesianVoter()
        assert agent.vote(clear_counts, rng) == 2

    def test_prior_matters_with_zero_counts(self, rng, zero_counts):
        """With no observations, posterior = prior (uniform), pick randomly."""
        agent = BayesianVoter()
        vote = agent.vote(zero_counts, rng)
        assert 0 <= vote < K_OPTIONS

    def test_different_from_majority_on_ambiguous(self):
        """Bayesian prior can break ties differently than raw counts."""
        # With counts [1, 0, 0, 0, 0], majority picks 0, Bayesian also picks 0
        # (prior 1 + count 1 = 2 vs 1 + 0 = 1), consistent but distinguishable
        counts = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
        rng = np.random.default_rng(42)
        assert BayesianVoter().vote(counts, rng) == 0


class TestCautiousVoter:
    def test_votes_when_confident(self, rng, clear_counts):
        agent = CautiousVoter(threshold=0.30)
        assert agent.vote(clear_counts, rng) == 2

    def test_abstains_when_uncertain(self, rng, zero_counts):
        """Zero counts -> uniform posterior (0.20 per option) < 0.30 threshold."""
        agent = CautiousVoter(threshold=0.30)
        assert agent.vote(zero_counts, rng) == -1

    def test_threshold_zero_never_abstains(self, rng, zero_counts):
        agent = CautiousVoter(threshold=0.0)
        assert agent.vote(zero_counts, rng) >= 0


class TestRandomByzantine:
    def test_ignores_signal(self, rng, clear_counts):
        agent = RandomByzantine()
        votes = set(agent.vote(clear_counts, rng) for _ in range(200))
        assert len(votes) > 1

    def test_votes_in_range(self, rng, clear_counts):
        agent = RandomByzantine()
        for _ in range(50):
            v = agent.vote(clear_counts, rng)
            assert 0 <= v < K_OPTIONS


class TestStrategicByzantine:
    def test_always_votes_zero(self, rng, clear_counts):
        agent = StrategicByzantine()
        for _ in range(20):
            assert agent.vote(clear_counts, rng) == 0


class TestMimickingByzantine:
    def test_sometimes_honest_sometimes_adversarial(self, rng, clear_counts):
        agent = MimickingByzantine(flip_prob=0.5)
        votes = [agent.vote(clear_counts, rng) for _ in range(200)]
        # Should have a mix: option 2 (honest) and option 0 (adversarial)
        assert any(v == 2 for v in votes)
        assert any(v == 0 for v in votes)

    def test_flip_prob_zero_always_honest(self, rng, clear_counts):
        agent = MimickingByzantine(flip_prob=0.0)
        for _ in range(20):
            assert agent.vote(clear_counts, rng) == 2


class TestFactories:
    def test_make_honest_agent_all_types(self):
        for name in HONEST_TYPES:
            agent = make_honest_agent(name)
            assert agent is not None

    def test_make_byzantine_agent_all_types(self):
        for name in BYZANTINE_TYPES:
            agent = make_byzantine_agent(name)
            assert agent is not None

    def test_unknown_honest_type_raises(self):
        with pytest.raises(KeyError):
            make_honest_agent("nonexistent")

    def test_unknown_byzantine_type_raises(self):
        with pytest.raises(KeyError):
            make_byzantine_agent("nonexistent")
