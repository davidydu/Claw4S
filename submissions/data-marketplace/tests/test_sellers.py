"""Tests for seller types."""

import numpy as np

from src.sellers import HonestSeller, StrategicSeller, PredatorySeller, create_seller


class TestHonestSeller:
    def test_claimed_equals_actual(self):
        s = HonestSeller(0, quality=0.8, rng=np.random.default_rng(0))
        offer = s.make_offer(0)
        assert offer.claimed_quality == offer.actual_quality == 0.8

    def test_price_proportional_to_quality(self):
        s1 = HonestSeller(0, quality=0.4, rng=np.random.default_rng(0))
        s2 = HonestSeller(1, quality=0.8, rng=np.random.default_rng(0))
        assert s1.make_offer(0).price < s2.make_offer(0).price

    def test_profit_tracking(self):
        s = HonestSeller(0, quality=0.5, rng=np.random.default_rng(0))
        s.record_sale(0.25)
        assert s.total_revenue == 0.25
        assert s.total_sales == 1
        assert s.profit == 0.25 - s.production_cost


class TestStrategicSeller:
    def test_over_claims_quality(self):
        s = StrategicSeller(0, quality=0.3, rng=np.random.default_rng(0))
        offer = s.make_offer(0)
        assert offer.claimed_quality > offer.actual_quality

    def test_adapts_over_time(self):
        s = StrategicSeller(0, quality=0.3, rng=np.random.default_rng(0))
        # Simulate 60 rounds with no sales
        for r in range(60):
            s.make_offer(r)
            s.record_no_sale()
        claim_after = s.make_offer(60).claimed_quality
        # Should have lowered claims
        assert claim_after <= 0.7 + 0.01  # started at 0.7, should go down


class TestPredatorySeller:
    def test_always_claims_max(self):
        s = PredatorySeller(0, quality=0.1, rng=np.random.default_rng(0))
        offer = s.make_offer(0)
        assert offer.claimed_quality == 1.0
        assert offer.actual_quality == 0.1

    def test_high_price_despite_low_quality(self):
        s = PredatorySeller(0, quality=0.1, rng=np.random.default_rng(0))
        offer = s.make_offer(0)
        # Price should be high (close to 0.5 * 0.95)
        assert offer.price > 0.4


class TestFactory:
    def test_create_all_types(self):
        rng = np.random.default_rng(0)
        for t in ["honest", "strategic", "predatory"]:
            s = create_seller(t, seller_id=0, quality=0.5, rng=rng)
            assert s.seller_type == t
