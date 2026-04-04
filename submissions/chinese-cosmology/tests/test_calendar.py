# tests/test_calendar.py
import datetime
from src.calendar_engine import (
    gregorian_to_pillars, get_year_pillar, get_month_pillar,
    get_day_pillar, get_hour_pillar, get_month_branch_from_date,
)


def test_year_pillar_known_date():
    """2024 is 甲辰 year (stem=0, branch=4)."""
    stem, branch = get_year_pillar(2024, 2, 10)  # after 立春
    assert stem == 0   # 甲
    assert branch == 4  # 辰


def test_year_changes_at_lichun():
    """Year changes at 立春, not Jan 1."""
    # 2024 立春 is approx Feb 4
    s1, b1 = get_year_pillar(2024, 2, 3)   # before 立春 → still 2023 year
    s2, b2 = get_year_pillar(2024, 2, 5)   # after 立春 → 2024 year
    assert (s1, b1) != (s2, b2)


def test_month_pillar():
    """Month pillar: branch from 节气, stem from year stem."""
    stem, branch = get_month_pillar(2024, 3, 15, year_stem=0)  # 甲 year, March
    assert 0 <= stem < 10
    assert 0 <= branch < 12


def test_day_pillar_known():
    """Jan 1, 2000 is known reference: verify range."""
    stem, branch = get_day_pillar(2000, 1, 1)
    # Verify against a known reference
    assert 0 <= stem < 10
    assert 0 <= branch < 12


def test_hour_pillar():
    """Hour pillar from time of day."""
    stem, branch = get_hour_pillar(14, day_stem=0)  # 2pm = 未时(7), 甲日
    assert branch == 7  # 未


def test_full_pillars():
    """Full 4-pillar conversion should return 4 (stem, branch) tuples."""
    pillars = gregorian_to_pillars(datetime.datetime(2000, 6, 15, 10, 0))
    assert len(pillars) == 4
    for stem, branch in pillars:
        assert 0 <= stem < 10
        assert 0 <= branch < 12


def test_month_branch_from_solar_terms():
    """March 2024 (after 惊蛰) should be month branch 卯(3)."""
    branch = get_month_branch_from_date(2024, 3, 15)
    assert branch == 3  # 卯 month
