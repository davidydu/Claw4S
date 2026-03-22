"""Tests for src/game.py — CoordinationGame."""

import numpy as np
import pytest

from src.game import CoordinationGame


class TestPriorGeneration:
    """Test that priors are generated correctly for various disagreement levels."""

    def test_priors_shape(self):
        g = CoordinationGame(n_agents=4, n_actions=5, disagreement=0.0, seed=42)
        assert g.priors.shape == (4, 5)

    def test_priors_normalised(self):
        for d in [0.0, 0.3, 0.7, 1.0]:
            g = CoordinationGame(n_agents=4, n_actions=5, disagreement=d, seed=42)
            row_sums = g.priors.sum(axis=1)
            np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_zero_disagreement_all_same_preference(self):
        g = CoordinationGame(n_agents=4, n_actions=5, disagreement=0.0, seed=42)
        prefs = g.preferred_actions()
        assert len(set(prefs)) == 1, f"At d=0, all agents should prefer same action, got {prefs}"

    def test_high_disagreement_different_preferences(self):
        g = CoordinationGame(n_agents=4, n_actions=5, disagreement=1.0, seed=42)
        prefs = g.preferred_actions()
        # With 4 agents and 5 actions, at max disagreement not all should agree
        assert len(set(prefs)) > 1, f"At d=1.0, agents should disagree, got {prefs}"

    def test_agreement_score_zero_disagreement(self):
        g = CoordinationGame(n_agents=4, n_actions=5, disagreement=0.0, seed=42)
        assert g.agreement_score() == 1.0

    def test_deterministic_with_seed(self):
        g1 = CoordinationGame(n_agents=4, n_actions=5, disagreement=0.5, seed=99)
        g2 = CoordinationGame(n_agents=4, n_actions=5, disagreement=0.5, seed=99)
        np.testing.assert_array_equal(g1.priors, g2.priors)

    def test_different_seeds_different_priors(self):
        g1 = CoordinationGame(n_agents=4, n_actions=5, disagreement=0.5, seed=42)
        g2 = CoordinationGame(n_agents=4, n_actions=5, disagreement=0.5, seed=99)
        assert not np.allclose(g1.priors, g2.priors)


class TestPayoff:
    """Test the coordination payoff function."""

    def test_all_same_action_payoff_one(self):
        g = CoordinationGame(n_agents=4)
        payoffs = g.payoff([2, 2, 2, 2])
        np.testing.assert_array_equal(payoffs, np.ones(4))

    def test_different_actions_payoff_zero(self):
        g = CoordinationGame(n_agents=4)
        payoffs = g.payoff([0, 1, 2, 3])
        np.testing.assert_array_equal(payoffs, np.zeros(4))

    def test_partial_agreement_payoff_zero(self):
        g = CoordinationGame(n_agents=4)
        payoffs = g.payoff([1, 1, 1, 2])
        np.testing.assert_array_equal(payoffs, np.zeros(4))

    def test_three_agents(self):
        g = CoordinationGame(n_agents=3)
        assert np.all(g.payoff([0, 0, 0]) == 1.0)
        assert np.all(g.payoff([0, 0, 1]) == 0.0)


class TestEntropy:
    """Test prior entropy computation."""

    def test_peaked_prior_low_entropy(self):
        g = CoordinationGame(n_agents=1, n_actions=5, disagreement=0.0, seed=42)
        entropy = g.prior_entropy()
        assert entropy[0] < np.log(5)  # must be less than uniform entropy

    def test_entropy_non_negative(self):
        for d in [0.0, 0.5, 1.0]:
            g = CoordinationGame(n_agents=4, n_actions=5, disagreement=d, seed=42)
            assert np.all(g.prior_entropy() >= 0)
