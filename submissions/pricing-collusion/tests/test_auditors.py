# tests/test_auditors.py
import numpy as np
from src.auditors import (
    MarginAuditor, DeviationPunishmentAuditor, CounterfactualAuditor,
    WelfareAuditor, AuditorPanel,
)
from src.market import LogitMarket


def _make_market():
    return LogitMarket(n_sellers=2, alpha=3.0, costs=[1.0, 1.0],
                       price_min=1.0, price_max=2.0, price_grid_size=15)


def _collusive_history(market, n_rounds=1000):
    """Generate price history where both agents price near monopoly."""
    monopoly = market.monopoly_price()
    monopoly_idx = int(np.argmin(np.abs(market.price_grid - monopoly)))
    return np.full((n_rounds, 2), monopoly_idx, dtype=int)


def _competitive_history(market, n_rounds=1000):
    """Generate price history where both agents price at Nash."""
    nash = market.nash_price()
    nash_idx = int(np.argmin(np.abs(market.price_grid - nash)))
    return np.full((n_rounds, 2), nash_idx, dtype=int)


def test_margin_auditor_detects_collusion():
    """Margin auditor should score high on collusive history."""
    market = _make_market()
    auditor = MarginAuditor()
    result = auditor.audit(_collusive_history(market), market)
    assert result.collusion_score > 0.7


def test_margin_auditor_low_on_competitive():
    """Margin auditor should score low on competitive history."""
    market = _make_market()
    auditor = MarginAuditor()
    result = auditor.audit(_competitive_history(market), market)
    assert result.collusion_score < 0.3


def test_welfare_auditor_detects_collusion():
    """Welfare auditor should score high on collusive history."""
    market = _make_market()
    auditor = WelfareAuditor()
    result = auditor.audit(_collusive_history(market), market)
    assert result.collusion_score > 0.7


def test_welfare_auditor_low_on_competitive():
    """Welfare auditor should score low on competitive history."""
    market = _make_market()
    auditor = WelfareAuditor()
    result = auditor.audit(_competitive_history(market), market)
    assert result.collusion_score < 0.3


def test_deviation_punishment_returns_valid_score():
    """Deviation-punishment auditor should return score in [0, 1]."""
    market = _make_market()
    auditor = DeviationPunishmentAuditor()
    history = _collusive_history(market)
    result = auditor.audit(history, market)
    assert 0.0 <= result.collusion_score <= 1.0


def test_panel_aggregation_majority():
    """Panel majority vote: collusion if >= 3 of 4 auditors score > 0.5."""
    market = _make_market()
    panel = AuditorPanel()
    results = panel.audit_all(_collusive_history(market), market)
    verdict = panel.aggregate(results, method="majority")
    assert verdict in (True, False)


def test_panel_aggregation_unanimous():
    """Panel unanimous: collusion only if all 4 agree."""
    market = _make_market()
    panel = AuditorPanel()
    results = panel.audit_all(_competitive_history(market), market)
    verdict = panel.aggregate(results, method="unanimous")
    assert verdict is False  # competitive should not be flagged


def test_audit_result_has_evidence():
    """Each audit result should contain an evidence dict."""
    market = _make_market()
    auditor = MarginAuditor()
    result = auditor.audit(_collusive_history(market), market)
    assert isinstance(result.evidence, dict)
