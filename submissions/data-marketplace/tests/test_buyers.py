"""Tests for buyer types."""

import numpy as np

from src.sellers import Offer
from src.buyers import NaiveBuyer, ReputationBuyer, AnalyticalBuyer, create_buyer


def _make_offers():
    """Three offers: cheap-low, mid-high, expensive-max."""
    return [
        Offer(seller_id=0, seller_type="honest", price=0.1, claimed_quality=0.3, actual_quality=0.3),
        Offer(seller_id=1, seller_type="honest", price=0.3, claimed_quality=0.8, actual_quality=0.8),
        Offer(seller_id=2, seller_type="predatory", price=0.25, claimed_quality=1.0, actual_quality=0.1),
    ]


class TestNaiveBuyer:
    def test_prefers_cheap_high_claim(self):
        b = NaiveBuyer(0, n_states=5, rng=np.random.default_rng(0))
        offers = _make_offers()
        choice = b.choose_offer(offers, 0)
        # Offer 2 claims quality=1.0 and is cheaper than offer 1 (0.25 < 0.3)
        assert choice == 2

    def test_abstains_when_too_expensive(self):
        b = NaiveBuyer(0, n_states=5, rng=np.random.default_rng(0), budget_per_round=0.05)
        offers = _make_offers()
        choice = b.choose_offer(offers, 0)
        assert choice is None

    def test_belief_update(self):
        b = NaiveBuyer(0, n_states=3, rng=np.random.default_rng(0))
        old_belief = b.belief.copy()
        data = np.array([0, 0, 0, 1, 2])
        b.update_belief(data)
        # State 0 should have higher belief now
        assert b.belief[0] > old_belief[0]


class TestReputationBuyer:
    def test_updates_reputation(self):
        b = ReputationBuyer(0, n_states=5, rng=np.random.default_rng(0))
        b.update_reputation(seller_id=1, claimed_quality=0.8, experienced_quality=0.8)
        assert b._rep_score(1) == 1.0  # perfect match

    def test_penalises_over_claimers(self):
        b = ReputationBuyer(0, n_states=5, rng=np.random.default_rng(0))
        b.update_reputation(seller_id=1, claimed_quality=1.0, experienced_quality=0.2)
        assert b._rep_score(1) < 0.5

    def test_unknown_seller_gets_neutral(self):
        b = ReputationBuyer(0, n_states=5, rng=np.random.default_rng(0))
        assert b._rep_score(99) == 0.5


class TestAnalyticalBuyer:
    def test_quality_estimate_high_for_similar(self):
        b = AnalyticalBuyer(0, n_states=3, rng=np.random.default_rng(0))
        data = np.array([0, 0, 0, 1, 1, 2])
        obs = np.array([0, 0, 0, 1, 1, 2])
        b.update_quality_estimate(seller_id=0, data_samples=data, observation_samples=obs)
        est = b._estimated_quality(0)
        assert est > 0.8

    def test_quality_estimate_low_for_different(self):
        b = AnalyticalBuyer(0, n_states=5, rng=np.random.default_rng(0))
        data = np.array([0, 0, 0, 0, 0])
        obs = np.array([4, 4, 4, 4, 4])
        b.update_quality_estimate(seller_id=0, data_samples=data, observation_samples=obs)
        est = b._estimated_quality(0)
        assert est < 0.7


class TestFactory:
    def test_create_all_types(self):
        rng = np.random.default_rng(0)
        for t in ["naive", "reputation", "analytical"]:
            b = create_buyer(t, buyer_id=0, n_states=5, rng=rng)
            assert b.buyer_type == t
