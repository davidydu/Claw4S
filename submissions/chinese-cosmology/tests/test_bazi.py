# tests/test_bazi.py
import datetime
from src.bazi import BaZiAgent


def test_element_count():
    """Should count all 5 elements across pillars."""
    agent = BaZiAgent()
    result = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
    elements = result["element_counts"]
    assert set(elements.keys()) == {"wood", "fire", "earth", "metal", "water"}
    assert sum(elements.values()) > 0


def test_day_master():
    """Day master should be the day stem's element."""
    agent = BaZiAgent()
    result = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
    assert result["day_master"] in {"wood", "fire", "earth", "metal", "water"}


def test_day_master_strength():
    """Strength should be between 0 and 1."""
    agent = BaZiAgent()
    result = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
    assert 0.0 <= result["day_master_strength"] <= 1.0


def test_ten_gods():
    """Should identify 十神 for each pillar position."""
    agent = BaZiAgent()
    result = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
    gods = result["ten_gods"]
    assert len(gods) > 0
    valid_gods = {"比肩", "劫财", "食神", "伤官", "偏财", "正财", "七杀", "正官", "偏印", "正印"}
    for god in gods.values():
        assert god in valid_gods


def test_domain_scores():
    """Should produce 5 domain scores in [0, 1]."""
    agent = BaZiAgent()
    result = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
    scores = result["domain_scores"]
    assert set(scores.keys()) == {"career", "wealth", "relationships", "health", "overall"}
    for v in scores.values():
        assert 0.0 <= v <= 1.0


def test_reproducible():
    """Same datetime → same result."""
    agent = BaZiAgent()
    r1 = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
    r2 = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
    assert r1["domain_scores"] == r2["domain_scores"]


def test_different_datetimes_differ():
    """Different birth times should generally produce different scores."""
    agent = BaZiAgent()
    r1 = agent.analyze(datetime.datetime(2000, 1, 1, 0, 0))
    r2 = agent.analyze(datetime.datetime(2000, 7, 15, 14, 0))
    assert r1["domain_scores"] != r2["domain_scores"]
