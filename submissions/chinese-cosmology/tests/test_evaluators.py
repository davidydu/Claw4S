# tests/test_evaluators.py
import numpy as np
from src.evaluators import (
    CorrelationEvaluator, MutualInformationEvaluator,
    DomainAgreementEvaluator, WuXingPredictivenessEvaluator,
    EvaluatorPanel, BaseEvaluator,
)


def _correlated_scores(n=100, seed=42):
    """Two systems that agree: scores are correlated."""
    rng = np.random.default_rng(seed)
    bazi = rng.uniform(0, 1, n)
    ziwei = bazi + rng.normal(0, 0.1, n)
    ziwei = np.clip(ziwei, 0, 1)
    return bazi, ziwei


def _uncorrelated_scores(n=100, seed=42):
    """Two systems that disagree: scores are independent."""
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, n), rng.uniform(0, 1, n)


def test_correlation_high():
    bazi, ziwei = _correlated_scores()
    ev = CorrelationEvaluator()
    result = ev.evaluate(bazi, ziwei)
    assert result.consistency_score > 0.7


def test_correlation_low():
    bazi, ziwei = _uncorrelated_scores()
    ev = CorrelationEvaluator()
    result = ev.evaluate(bazi, ziwei)
    assert result.consistency_score < 0.4


def test_mutual_info_correlated():
    bazi, ziwei = _correlated_scores()
    ev = MutualInformationEvaluator()
    result = ev.evaluate(bazi, ziwei)
    assert result.consistency_score > 0.3


def test_domain_agreement_high():
    bazi, ziwei = _correlated_scores()
    ev = DomainAgreementEvaluator()
    result = ev.evaluate(bazi, ziwei)
    assert result.consistency_score > 0.7


def test_domain_agreement_low():
    bazi, ziwei = _uncorrelated_scores()
    ev = DomainAgreementEvaluator()
    result = ev.evaluate(bazi, ziwei)
    assert result.consistency_score < 0.7


def test_wuxing_predictiveness():
    rng = np.random.default_rng(42)
    wuxing = rng.uniform(0, 1, 100)
    target = wuxing + rng.normal(0, 0.1, 100)
    ev = WuXingPredictivenessEvaluator()
    result = ev.evaluate(wuxing, np.clip(target, 0, 1))
    assert 0.0 <= result.consistency_score <= 1.0


def test_panel():
    panel = EvaluatorPanel()
    bazi, ziwei = _correlated_scores()
    wuxing = bazi * 0.8 + 0.1
    results = panel.evaluate_all(bazi, ziwei, wuxing)
    assert len(results) == 4


def test_eval_result_has_evidence():
    ev = CorrelationEvaluator()
    bazi, ziwei = _correlated_scores()
    result = ev.evaluate(bazi, ziwei)
    assert isinstance(result.evidence, dict)
