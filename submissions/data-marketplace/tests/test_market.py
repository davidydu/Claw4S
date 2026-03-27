"""Tests for DataMarketplace."""

import numpy as np

from src.environment import DataEnvironment
from src.sellers import HonestSeller, PredatorySeller
from src.buyers import NaiveBuyer, ReputationBuyer, AnalyticalBuyer
from src.market import DataMarketplace, Transaction


def _make_market(info_regime="opaque", seed=42):
    """Create a small test market: 1 honest (medium-q) + 1 predatory seller, 1 naive buyer.

    Honest seller quality=0.5, price=0.25, claimed_quality=0.5 (below 0.7 threshold).
    Predatory seller quality=0.1, price=0.475, claimed_quality=1.0 (above threshold).
    In opaque mode, naive buyer will prefer the predatory seller (high claimed quality).
    """
    rng = np.random.default_rng(seed)
    env = DataEnvironment(n_states=5, rng=rng)
    sellers = [
        HonestSeller(0, quality=0.5, rng=np.random.default_rng(seed + 1)),
        PredatorySeller(1, quality=0.1, rng=np.random.default_rng(seed + 2)),
    ]
    buyers = [NaiveBuyer(0, n_states=5, rng=np.random.default_rng(seed + 3))]
    return DataMarketplace(env, sellers, buyers, info_regime=info_regime)


class TestMarketBasic:
    def test_single_round_produces_transaction(self):
        m = _make_market()
        txns = m.run_round(0)
        assert len(txns) == 1  # 1 buyer, 1 purchase

    def test_transaction_fields(self):
        m = _make_market()
        txns = m.run_round(0)
        t = txns[0]
        assert isinstance(t, Transaction)
        assert t.round_idx == 0
        assert t.price > 0
        assert 0 <= t.actual_quality <= 1

    def test_multiple_rounds(self):
        m = _make_market()
        txns = m.run(50)
        assert len(txns) == 50  # 1 buyer buys each round

    def test_naive_buyer_exploited_by_predatory(self):
        """Naive buyer should buy from predatory seller (claims q=1.0, only high-claimer)."""
        m = _make_market(seed=0)
        m.run(100)
        pred_txns = [t for t in m.transactions if t.seller_type == "predatory"]
        # Honest seller claims 0.5 (< 0.7 threshold), so naive always picks predatory
        assert len(pred_txns) >= 90


class TestMarketTransparent:
    def test_transparent_prevents_exploitation(self):
        """In transparent regime, predatory seller's true quality is exposed.

        Honest q=0.5 (below threshold), Predatory actual q=0.1 (below threshold).
        In transparent mode, buyer sees actual qualities, neither qualifies as
        'high quality', so buyer falls through to cheapest-affordable (honest at 0.25).
        """
        m = _make_market(info_regime="transparent", seed=0)
        m.run(100)
        honest_txns = [t for t in m.transactions if t.seller_type == "honest"]
        # In transparent, predatory q=0.1 (claimed becomes 0.1, below threshold)
        # Honest q=0.5 (claimed becomes 0.5, also below threshold)
        # Fallback: cheapest affordable → honest at price 0.25
        assert len(honest_txns) >= 90


class TestMetrics:
    def test_price_quality_correlation_honest_market(self):
        """With reputation buyers exploring across honest sellers, price-quality correlation is high."""
        rng = np.random.default_rng(0)
        env = DataEnvironment(n_states=5, rng=rng)
        sellers = [
            HonestSeller(0, quality=0.3, rng=np.random.default_rng(1)),
            HonestSeller(1, quality=0.6, rng=np.random.default_rng(2)),
            HonestSeller(2, quality=0.9, rng=np.random.default_rng(3)),
        ]
        buyers = [
            # Reputation buyers explore, creating transactions across sellers
            ReputationBuyer(0, n_states=5, rng=np.random.default_rng(4)),
            ReputationBuyer(1, n_states=5, rng=np.random.default_rng(5)),
        ]
        m = DataMarketplace(env, sellers, buyers, info_regime="opaque")
        m.run(500)
        # With exploration, we get txns from multiple sellers
        unique_sellers = set(t.seller_id for t in m.transactions)
        assert len(unique_sellers) >= 2
        # All honest → price correlates with quality where there's variation
        corr = m.price_quality_correlation()
        assert corr > 0.3

    def test_lemons_index_zero_for_honest(self):
        rng = np.random.default_rng(0)
        env = DataEnvironment(n_states=5, rng=rng)
        sellers = [HonestSeller(0, quality=0.9, rng=np.random.default_rng(1))]
        buyers = [NaiveBuyer(0, n_states=5, rng=np.random.default_rng(2))]
        m = DataMarketplace(env, sellers, buyers)
        m.run(50)
        assert m.lemons_index() == 0.0  # no low-quality sellers

    def test_market_efficiency_bounded(self):
        m = _make_market()
        m.run(100)
        eff = m.market_efficiency()
        assert 0 <= eff <= 1.0
