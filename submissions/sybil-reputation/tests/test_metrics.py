"""Tests for evaluation metrics."""

from src.agents import Agent
from src.metrics import (
    reputation_accuracy,
    sybil_detection_rate,
    honest_welfare,
    market_efficiency,
    compute_all_metrics,
)


def _honest(aid, quality):
    return Agent(agent_id=aid, true_quality=quality)


def _sybil(aid):
    return Agent(agent_id=aid, true_quality=0.1, is_sybil=True)


def test_reputation_accuracy_perfect():
    agents = [_honest(0, 0.2), _honest(1, 0.5), _honest(2, 0.8)]
    scores = {0: 0.2, 1: 0.5, 2: 0.8}  # Perfect match
    acc = reputation_accuracy(agents, scores)
    assert acc > 0.9


def test_reputation_accuracy_inverted():
    agents = [_honest(0, 0.2), _honest(1, 0.5), _honest(2, 0.8)]
    scores = {0: 0.8, 1: 0.5, 2: 0.2}  # Inverted
    acc = reputation_accuracy(agents, scores)
    assert acc < -0.5


def test_sybil_detection_no_sybils():
    honest = [_honest(0, 0.5)]
    rate = sybil_detection_rate(honest, [], {0: 0.5})
    assert rate == 1.0


def test_sybil_detection_all_detected():
    honest = [_honest(0, 0.5), _honest(1, 0.6)]
    sybils = [_sybil(2), _sybil(3)]
    scores = {0: 0.7, 1: 0.8, 2: 0.1, 3: 0.2}
    rate = sybil_detection_rate(honest, sybils, scores)
    assert rate == 1.0


def test_sybil_detection_none_detected():
    honest = [_honest(0, 0.5), _honest(1, 0.6)]
    sybils = [_sybil(2), _sybil(3)]
    scores = {0: 0.3, 1: 0.4, 2: 0.9, 3: 0.8}
    rate = sybil_detection_rate(honest, sybils, scores)
    assert rate == 0.0


def test_honest_welfare():
    agents = [_honest(0, 0.5), _honest(1, 0.5)]
    scores = {0: 0.6, 1: 0.8}
    w = honest_welfare(agents, scores)
    assert abs(w - 0.7) < 1e-6


def test_market_efficiency_perfect():
    agents = [_honest(0, 0.1), _honest(1, 0.5), _honest(2, 0.9)]
    scores = {0: 0.1, 1: 0.5, 2: 0.9}
    eff = market_efficiency(agents, scores)
    assert eff > 0.9


def test_compute_all_metrics_keys():
    honest = [_honest(0, 0.5), _honest(1, 0.6), _honest(2, 0.7)]
    sybils = [_sybil(3)]
    scores = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.1}
    m = compute_all_metrics(honest, sybils, scores)
    assert set(m.keys()) == {
        "reputation_accuracy",
        "sybil_detection_rate",
        "honest_welfare",
        "market_efficiency",
    }
