"""Tests for market auditors."""

import numpy as np

from src.environment import DataEnvironment
from src.sellers import HonestSeller, PredatorySeller
from src.buyers import NaiveBuyer, ReputationBuyer
from src.market import DataMarketplace
from src.auditors import (
    FairPricingAuditor,
    ExploitationAuditor,
    MarketEfficiencyAuditor,
    InformationAsymmetryAuditor,
    AuditPanel,
)


def _honest_market(n_rounds=100, seed=42):
    rng = np.random.default_rng(seed)
    env = DataEnvironment(n_states=5, rng=rng)
    sellers = [HonestSeller(0, quality=0.9, rng=np.random.default_rng(seed + 1))]
    buyers = [NaiveBuyer(0, n_states=5, rng=np.random.default_rng(seed + 2))]
    m = DataMarketplace(env, sellers, buyers, info_regime="opaque")
    m.run(n_rounds)
    return m


def _predatory_market(n_rounds=100, seed=42):
    rng = np.random.default_rng(seed)
    env = DataEnvironment(n_states=5, rng=rng)
    sellers = [PredatorySeller(0, quality=0.1, rng=np.random.default_rng(seed + 1))]
    buyers = [NaiveBuyer(0, n_states=5, rng=np.random.default_rng(seed + 2))]
    m = DataMarketplace(env, sellers, buyers, info_regime="opaque")
    m.run(n_rounds)
    return m


class TestFairPricingAuditor:
    def test_score_bounded(self):
        m = _honest_market()
        r = FairPricingAuditor().audit(m)
        assert 0.0 <= r.score <= 1.0

    def test_has_details(self):
        m = _honest_market()
        r = FairPricingAuditor().audit(m)
        assert "pearson_r" in r.details


class TestExploitationAuditor:
    def test_honest_market_low_exploitation(self):
        m = _honest_market()
        r = ExploitationAuditor().audit(m)
        assert r.score > 0.8  # honest seller → little exploitation

    def test_predatory_market_high_exploitation(self):
        m = _predatory_market()
        r = ExploitationAuditor().audit(m)
        assert r.score < 0.5  # predatory → lots of exploitation


class TestMarketEfficiencyAuditor:
    def test_honest_market_efficient(self):
        m = _honest_market()
        r = MarketEfficiencyAuditor().audit(m)
        assert r.score > 0.0
        assert "total_surplus" in r.details


class TestInformationAsymmetryAuditor:
    def test_honest_low_asymmetry(self):
        m = _honest_market()
        r = InformationAsymmetryAuditor().audit(m)
        assert r.score > 0.9  # honest → claims match reality

    def test_predatory_high_asymmetry(self):
        m = _predatory_market()
        r = InformationAsymmetryAuditor().audit(m)
        assert r.score < 0.3  # predatory claims q=1.0 but delivers q=0.1


class TestAuditPanel:
    def test_runs_all_four(self):
        m = _honest_market()
        panel = AuditPanel()
        results = panel.audit(m)
        assert len(results) == 4

    def test_summary_keys(self):
        m = _honest_market()
        panel = AuditPanel()
        results = panel.audit(m)
        summary = panel.summary(results)
        assert set(summary.keys()) == {"fair_pricing", "exploitation", "market_efficiency", "information_asymmetry"}
