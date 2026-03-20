# src/experiment.py
"""Experiment runner for the Chinese cosmology analysis.

Generates a corpus of birth datetimes spanning a 60-year 甲子 cycle and
runs all three system agents (BaZi, Zi Wei Dou Shu, Wu Xing) on each.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass

from src.bazi import BaZiAgent
from src.ziwei import ZiWeiAgent
from src.wuxing import WuXingAgent


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    """Configuration for the full experiment.

    Attributes:
        start_year: first year to include (inclusive); defaults to 1984
        end_year:   last year to include (exclusive); defaults to 2044
    """
    start_year: int = 1984
    end_year: int = 2044


# ---------------------------------------------------------------------------
# Chart corpus builder
# ---------------------------------------------------------------------------

def build_chart_configs(start_year: int = 1984, end_year: int = 2044) -> list:
    """Build a list of birth datetimes at 2-hour intervals (one per 时辰).

    Generates one datetime per 时辰 (12 per day) for every day in the
    specified year range.  The datetime corresponds to the start of each
    2-hour time period:
      子 → 23:00, 丑 → 01:00, 寅 → 03:00, ..., 亥 → 21:00

    Args:
        start_year: first calendar year to include (Jan 1)
        end_year:   first calendar year to exclude (stop at Dec 31 of end_year-1)

    Returns:
        list of datetime.datetime objects (naive)
    """
    # 时辰 start hours (24h clock): 子=23, 丑=1, 寅=3, 卯=5, 辰=7, 巳=9,
    #                                午=11, 未=13, 申=15, 酉=17, 戌=19, 亥=21
    # We use the "start of the 时辰" convention: 子 starts at 23:00 the previous
    # evening, but for day-of-birth labeling we use 01:00 for 丑, 03:00 for 寅 etc.
    # To keep it simple, use the even-hour start: 0,2,4,...,22 (local time).
    SHICHEN_HOURS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

    configs = []
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 1, 1)  # exclusive

    current = start_date
    one_day = datetime.timedelta(days=1)

    while current < end_date:
        for hour in SHICHEN_HOURS:
            configs.append(datetime.datetime(current.year, current.month,
                                             current.day, hour, 0))
        current += one_day

    return configs


# ---------------------------------------------------------------------------
# Single-chart analysis
# ---------------------------------------------------------------------------

def run_chart_analysis(dt: datetime.datetime) -> dict:
    """Run all three system agents on a single birth datetime.

    Args:
        dt: birth datetime

    Returns:
        dict with keys 'bazi', 'ziwei', 'wuxing', each containing the
        respective agent's analysis result dict (including 'domain_scores').
        Also includes 'datetime' as an ISO-format string.
    """
    bazi_agent = BaZiAgent()
    ziwei_agent = ZiWeiAgent()
    wuxing_agent = WuXingAgent()

    bazi_result = bazi_agent.analyze(dt)

    ziwei_result = ziwei_agent.analyze(dt)

    # Wu Xing takes the BaZi element counts as input
    wuxing_result = wuxing_agent.analyze(bazi_result["element_counts"])

    return {
        "datetime": dt.isoformat(),
        "bazi": bazi_result,
        "ziwei": ziwei_result,
        "wuxing": wuxing_result,
    }
