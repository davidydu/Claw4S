# tests/test_ziwei.py
"""Tests for the Zi Wei Dou Shu agent (紫微斗数)."""

import datetime
from src.ziwei import ZiWeiAgent


# Shared fixture datetime used in most tests
_DT = datetime.datetime(2000, 6, 15, 10, 0)


def test_life_palace():
    """命宫 should be a valid palace index 0-11."""
    agent = ZiWeiAgent()
    result = agent.analyze(_DT)
    assert 0 <= result["life_palace"] < 12


def test_five_element_bureau():
    """五行局 should be one of 2,3,4,5,6."""
    agent = ZiWeiAgent()
    result = agent.analyze(_DT)
    assert result["bureau"] in {2, 3, 4, 5, 6}


def test_star_placement():
    """All 14 major stars should be placed in palaces 0-11."""
    agent = ZiWeiAgent()
    result = agent.analyze(_DT)
    stars = result["star_placement"]
    assert len(stars) == 14
    for star, palace in stars.items():
        assert 0 <= palace < 12, f"{star} placed at invalid palace {palace}"


def test_palace_scores():
    """Each of the 12 palaces should have a score in [0, 1]."""
    agent = ZiWeiAgent()
    result = agent.analyze(_DT)
    scores = result["palace_scores"]
    assert len(scores) == 12
    for palace_name, score in scores.items():
        assert 0.0 <= score <= 1.0, (
            f"Palace '{palace_name}' score {score} out of [0,1]"
        )


def test_domain_scores():
    """Should produce exactly 5 domain scores, all in [0, 1]."""
    agent = ZiWeiAgent()
    result = agent.analyze(_DT)
    scores = result["domain_scores"]
    assert set(scores.keys()) == {"career", "wealth", "relationships", "health", "overall"}
    for domain, v in scores.items():
        assert 0.0 <= v <= 1.0, f"Domain '{domain}' score {v} out of [0,1]"


def test_brightness_scoring():
    """Palace scores should not all equal 0.5 (neutral sigmoid baseline).

    If all stars had zero contribution, sigmoid(0) = 0.5 for every palace.
    Real brightness-weighted scoring must produce some variation.
    """
    agent = ZiWeiAgent()
    result = agent.analyze(_DT)
    scores = list(result["palace_scores"].values())
    assert not all(s == 0.5 for s in scores), (
        "All palace scores are 0.5 — brightness scoring is not working"
    )


def test_reproducible():
    """Same datetime should always produce identical results."""
    agent = ZiWeiAgent()
    r1 = agent.analyze(_DT)
    r2 = agent.analyze(_DT)
    assert r1["domain_scores"] == r2["domain_scores"]
    assert r1["star_placement"] == r2["star_placement"]
    assert r1["palace_scores"] == r2["palace_scores"]
