# tests/test_wuxing.py
import numpy as np
from src.wuxing import WuXingAgent


def test_balanced_input_high_equilibrium():
    """Equal element counts → high equilibrium score."""
    agent = WuXingAgent()
    counts = {"wood": 2, "fire": 2, "earth": 2, "metal": 2, "water": 2}
    result = agent.analyze(counts)
    assert result["equilibrium_score"] > 0.7


def test_imbalanced_input_lower_equilibrium():
    """Dominated by one element → lower equilibrium score."""
    agent = WuXingAgent()
    counts = {"wood": 8, "fire": 1, "earth": 0, "metal": 0, "water": 1}
    result = agent.analyze(counts)
    assert result["equilibrium_score"] < 0.7


def test_stability_score():
    """Stability score should be in [0, 1]."""
    agent = WuXingAgent()
    counts = {"wood": 3, "fire": 2, "earth": 2, "metal": 1, "water": 2}
    result = agent.analyze(counts)
    assert 0.0 <= result["stability_score"] <= 1.0


def test_dominant_element():
    """Should identify the dominant element at equilibrium."""
    agent = WuXingAgent()
    counts = {"wood": 8, "fire": 1, "earth": 0, "metal": 0, "water": 1}
    result = agent.analyze(counts)
    assert result["dominant_element"] in {"wood", "fire", "earth", "metal", "water"}


def test_domain_scores():
    """Should produce 5 domain scores in [0, 1]."""
    agent = WuXingAgent()
    counts = {"wood": 3, "fire": 2, "earth": 2, "metal": 1, "water": 2}
    result = agent.analyze(counts)
    scores = result["domain_scores"]
    assert set(scores.keys()) == {"career", "wealth", "relationships", "health", "overall"}
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_convergence():
    """System should converge within max_steps."""
    agent = WuXingAgent()
    counts = {"wood": 3, "fire": 2, "earth": 2, "metal": 1, "water": 2}
    result = agent.analyze(counts)
    assert result["converged"] is True
