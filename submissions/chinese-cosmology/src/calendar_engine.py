# src/calendar_engine.py
"""Chinese calendar engine: Gregorian → 天干地支 pillar conversion.

Reference epoch: Jan 1, 1900 = 甲戌 day (stem=0, branch=10).
Year cycle anchor: 1984 = 甲子 year (stem=0, branch=0).
"""

import datetime
from src.tables import (
    FIVE_TIGERS, FIVE_RATS,
    SOLAR_TERMS, JIE_TO_MONTH_BRANCH,
)


def gregorian_to_pillars(dt):
    """Convert a datetime to 4 Chinese calendar pillars.

    Args:
        dt: datetime.datetime object

    Returns:
        List of 4 (stem, branch) tuples:
        [(year_stem, year_branch), (month_stem, month_branch),
         (day_stem, day_branch), (hour_stem, hour_branch)]
        All stems in range [0, 9], all branches in range [0, 11].
    """
    year_stem, year_branch = get_year_pillar(dt.year, dt.month, dt.day)
    month_stem, month_branch = get_month_pillar(
        dt.year, dt.month, dt.day, year_stem
    )
    day_stem, day_branch = get_day_pillar(dt.year, dt.month, dt.day)
    hour_stem, hour_branch = get_hour_pillar(dt.hour, day_stem)
    return [
        (year_stem, year_branch),
        (month_stem, month_branch),
        (day_stem, day_branch),
        (hour_stem, hour_branch),
    ]


def get_year_pillar(year, month, day):
    """Get year stem and branch.

    The Chinese year begins at 立春 (approx Feb 4), not Jan 1.
    Cycle anchor: 1984 = 甲子 (stem=0, branch=0).

    Args:
        year: Gregorian year (int)
        month: Gregorian month (int)
        day: Gregorian day (int)

    Returns:
        (stem, branch) tuple — stem in [0,9], branch in [0,11]
    """
    lichun = _get_lichun_date(year)
    effective_year = year if datetime.date(year, month, day) >= lichun else year - 1
    offset = effective_year - 1984
    stem = offset % 10
    branch = offset % 12
    return stem, branch


def get_month_pillar(year, month, day, year_stem):
    """Get month stem and branch.

    Month boundaries follow the 12 "节" (jie) solar terms.
    Stem is derived via the 五虎遁 (Five Tigers) rule.

    Args:
        year: Gregorian year (int)
        month: Gregorian month (int)
        day: Gregorian day (int)
        year_stem: stem index of the year pillar [0,9]

    Returns:
        (stem, branch) tuple — stem in [0,9], branch in [0,11]
    """
    month_branch = get_month_branch_from_date(year, month, day)
    # 五虎遁: get the stem of the 寅(2) month for this year stem
    yin_stem = FIVE_TIGERS[year_stem]
    # Count months from 寅(2) to the current month branch (mod 12)
    offset = (month_branch - 2) % 12
    month_stem = (yin_stem + offset) % 10
    return month_stem, month_branch


def get_month_branch_from_date(year, month, day):
    """Determine month branch from the 节气 solar term boundaries.

    Uses the SOLAR_TERMS table to find which jie term the date falls in.
    The 12 jie terms and their month branches:
      小寒→丑(1), 立春→寅(2), 惊蛰→卯(3), 清明→辰(4),
      立夏→巳(5), 芒种→午(6), 小暑→未(7), 立秋→申(8),
      白露→酉(9), 寒露→戌(10), 立冬→亥(11), 大雪→子(0)

    Args:
        year: Gregorian year (int)
        month: Gregorian month (int)
        day: Gregorian day (int)

    Returns:
        month branch index [0,11]
    """
    target = datetime.date(year, month, day)

    # Build a sorted list of (date, branch) for the current year and adjacent years
    # We need to look at year-1 through year+1 to handle Jan/Feb edge cases
    all_boundaries = []
    for y in (year - 1, year, year + 1):
        if y not in SOLAR_TERMS:
            continue
        dates = SOLAR_TERMS[y]  # list of 12 (month, day) tuples
        for jie_idx, (m, d) in enumerate(dates):
            branch = JIE_TO_MONTH_BRANCH[jie_idx]
            all_boundaries.append((datetime.date(y, m, d), branch))

    # Sort by date
    all_boundaries.sort(key=lambda x: x[0])

    # Find the last boundary that is <= target
    result_branch = 1  # default: 丑 (Jan before 小寒 is still 丑 from prior year)
    for boundary_date, branch in all_boundaries:
        if boundary_date <= target:
            result_branch = branch
        else:
            break

    return result_branch


def get_day_pillar(year, month, day):
    """Get day stem and branch using a fixed reference-date calculation.

    Reference: Jan 1, 1900 = 甲戌 day (stem=0, branch=10).
    Verified: this is consistent with standard 万年历 tables.

    Args:
        year: Gregorian year (int)
        month: Gregorian month (int)
        day: Gregorian day (int)

    Returns:
        (stem, branch) tuple — stem in [0,9], branch in [0,11]
    """
    ref = datetime.date(1900, 1, 1)
    target = datetime.date(year, month, day)
    delta = (target - ref).days
    # Reference offsets: stem=0 (甲), branch=10 (戌) on 1900-01-01
    stem = (0 + delta) % 10
    branch = (10 + delta) % 12
    return stem, branch


def get_hour_pillar(hour, day_stem):
    """Get hour stem and branch from clock hour and day stem.

    Time-to-branch mapping (traditional 时辰, each = 2 hours):
      子(0): 23:00-01:00, 丑(1): 01:00-03:00, 寅(2): 03:00-05:00,
      卯(3): 05:00-07:00, 辰(4): 07:00-09:00, 巳(5): 09:00-11:00,
      午(6): 11:00-13:00, 未(7): 13:00-15:00, 申(8): 15:00-17:00,
      酉(9): 17:00-19:00, 戌(10): 19:00-21:00, 亥(11): 21:00-23:00

    Note: 子 hour spans midnight, so hour 23 → branch 0.

    Args:
        hour: clock hour [0, 23]
        day_stem: stem index of the day pillar [0, 9]

    Returns:
        (stem, branch) tuple — stem in [0,9], branch in [0,11]
    """
    # Map 0-23h to 时辰 branch index
    # 23:00-00:59 → 子(0), 01:00-02:59 → 丑(1), ..., 21:00-22:59 → 亥(11)
    if hour == 23:
        hour_branch = 0
    else:
        hour_branch = (hour + 1) // 2

    # 五鼠遁: get the stem of the 子 hour for this day stem
    zi_stem = FIVE_RATS[day_stem]
    hour_stem = (zi_stem + hour_branch) % 10
    return hour_stem, hour_branch


def _get_lichun_date(year):
    """Get 立春 date for a given year from SOLAR_TERMS table.

    立春 is jie-term index 1 in the table (month February).
    Falls back to Feb 4 if year is out of range.

    Args:
        year: Gregorian year (int)

    Returns:
        datetime.date for 立春 of that year
    """
    if year in SOLAR_TERMS:
        m, d = SOLAR_TERMS[year][1]  # index 1 = 立春
        return datetime.date(year, m, d)
    # Fallback for years outside 1984-2044
    return datetime.date(year, 2, 4)
