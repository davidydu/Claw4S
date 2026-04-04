# src/ziwei.py
"""Zi Wei Dou Shu (紫微斗数) agent.

Computes 命宫, 五行局, star placements, palace scores (庙旺利陷),
and 5 life domain scores from a birth datetime.

Algorithm overview
------------------
Step 1: Determine 命宫 branch index using birth month and hour.
Step 2: Determine 命宫 stem via 五虎遁, then look up 纳音五行 → 五行局 number.
Step 3: Place 紫微星 via the ZIWEI_PLACEMENT lookup table (day × bureau).
Step 4: Place remaining 13 major stars relative to 紫微 (紫微系)
        and relative to 天府 (天府系), where 天府 is derived from 命宫 and 紫微.
Step 5: Compute palace scores using star_base_value × brightness, sigmoid-normalized.
Step 6: Map palace scores to 5 life domain scores.
"""

import math
import datetime

from src.calendar_engine import (
    gregorian_to_pillars,
    get_month_branch_from_date,
    get_day_pillar,
)
from src.tables import (
    HEAVENLY_STEMS,
    EARTHLY_BRANCHES,
    NAYIN_ELEMENTS,
    FIVE_TIGERS,
)
from src.ziwei_tables import (
    ZIWEI_PLACEMENT,
    ZIWEI_SERIES_OFFSETS,
    TIANFU_SERIES_OFFSETS,
    STAR_NATURE,
    STAR_BRIGHTNESS,
    DOMAIN_PALACE_OFFSET,
    PALACES,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Sigmoid function: maps any real → (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def _hour_to_branch(hour: int) -> int:
    """Convert 24-hour clock hour to 时辰 (time period) branch index.

    时辰 boundaries (branch → hours):
      子(0): 23-01, 丑(1): 01-03, 寅(2): 03-05, 卯(3): 05-07,
      辰(4): 07-09, 巳(5): 09-11, 午(6): 11-13, 未(7): 13-15,
      申(8): 15-17, 酉(9): 17-19, 戌(10): 19-21, 亥(11): 21-23

    Hour 23 maps to 子(0) of the NEXT day in strict terms, but for
    birth charts we treat the 2-hour block containing the hour.
    """
    # Each branch covers 2 hours starting from 23:00 for 子
    # Shift by 1 hour so that 23:00 → 子(0), 01:00 → 丑(1), etc.
    return ((hour + 1) % 24) // 2


def _get_lunar_day(dt: datetime.datetime) -> int:
    """Estimate lunar day (1-30) from a Gregorian datetime.

    For Zi Wei Dou Shu we need the lunar birth day. We use a simplified
    approximation based on the synodic month (29.53059 days).

    Reference new moon: 2000-01-06 18:14 UTC (known new moon)
    This gives a day-within-month estimate that is sufficient for the
    30×5 placement table.

    Returns an integer in [1, 30].
    """
    # Reference: known new moon at 2000-01-06.759 (UTC fraction)
    REF_NEW_MOON = datetime.datetime(2000, 1, 6, 18, 14)
    SYNODIC_MONTH = 29.530589  # days

    delta_days = (dt - REF_NEW_MOON).total_seconds() / 86400.0
    # Days since reference new moon, mod one synodic month
    phase_days = delta_days % SYNODIC_MONTH
    # Lunar day: 0 phase = new moon = day 1 of lunar month
    lunar_day = int(phase_days) + 1
    # Clamp to [1, 30]
    if lunar_day < 1:
        lunar_day = 1
    if lunar_day > 30:
        lunar_day = 30
    return lunar_day


def _get_nayin_index(stem: int, branch: int) -> int:
    """Get the 六十甲子 cycle index from stem and branch.

    The 60-cycle index maps to the NAYIN_ELEMENTS array (0-59).
    """
    # The cycle index satisfies: stem = idx % 10, branch = idx % 12
    # There is a unique index in [0, 59] for each (stem, branch) pair.
    for idx in range(60):
        if idx % 10 == stem and idx % 12 == branch:
            return idx
    raise ValueError(f"No 六十甲子 index for stem={stem}, branch={branch}")


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class ZiWeiAgent:
    """Zi Wei Dou Shu agent.

    Usage:
        agent = ZiWeiAgent()
        result = agent.analyze(datetime.datetime(2000, 6, 15, 10, 0))
        # result keys: life_palace, bureau, star_placement,
        #              palace_scores, domain_scores
    """

    def analyze(self, dt: datetime.datetime) -> dict:
        """Analyze a birth datetime with Zi Wei Dou Shu.

        Args:
            dt: Birth datetime (naive, local time assumed).

        Returns:
            dict with keys:
              - life_palace (int): 命宫 branch index [0-11]
              - bureau (int): 五行局 number {2, 3, 4, 5, 6}
              - star_placement (dict): {star_name: palace_branch_index}
              - palace_scores (dict): {palace_name: float [0,1]}
              - domain_scores (dict): {domain: float [0,1]}
        """
        # Step 1: Determine 命宫 branch
        life_palace_branch = self._get_life_palace_branch(dt)

        # Step 2: Determine 命宫 stem and 五行局
        life_palace_stem = self._get_life_palace_stem(dt, life_palace_branch)
        bureau = self._get_bureau(life_palace_stem, life_palace_branch)

        # Step 3: Place 紫微星
        lunar_day = _get_lunar_day(dt)
        ziwei_palace = ZIWEI_PLACEMENT[(lunar_day, bureau)]

        # Step 4: Place all 14 major stars
        star_placement = self._place_stars(ziwei_palace, life_palace_branch)

        # Step 5: Score each palace
        palace_scores = self._score_palaces(star_placement, life_palace_branch)

        # Step 6: Map to domain scores
        domain_scores = self._compute_domain_scores(palace_scores, life_palace_branch)

        return {
            "life_palace": life_palace_branch,
            "bureau": bureau,
            "star_placement": star_placement,
            "palace_scores": palace_scores,
            "domain_scores": domain_scores,
        }

    # -----------------------------------------------------------------------
    # Step 1: 命宫 branch
    # -----------------------------------------------------------------------

    def _get_life_palace_branch(self, dt: datetime.datetime) -> int:
        """Determine 命宫 branch index.

        Formula: (month_number + 1 - hour_index) mod 12
        Where:
          - month_number is the Chinese month branch (1-12 mapped to branch 2-1)
            but here we use the 1-indexed month number directly:
            month_number = (month_branch - 2) % 12 + 1  (i.e. 寅=1, 卯=2, ..., 丑=12)
          - hour_index: 子=0, 丑=1, ..., 亥=11

        Equivalently, using the raw branch indices:
          命宫_branch = (month_branch + 2 - hour_branch) % 12 + ...

        Standard formula from spec:
          命宫_branch_index = (month_number + 1 - hour_index) mod 12
          where month 1 = 寅(2), so month_number = (month_branch - 2) % 12 + 1
          and hour 子=0

        Simplification: month_number = (month_branch - 2) % 12 + 1
          命宫_branch = (((month_branch - 2) % 12 + 1) + 1 - hour_branch) % 12
                      = (month_branch - hour_branch) % 12
        """
        month_branch = get_month_branch_from_date(dt.year, dt.month, dt.day)
        hour_branch = _hour_to_branch(dt.hour)

        # Spec formula: (month_number + 1 - hour_index) mod 12
        # month_number is 1-indexed starting from 寅(2):
        #   寅 → month 1, 卯 → month 2, ..., 丑 → month 12
        month_number = (month_branch - 2) % 12 + 1
        life_palace_branch = (month_number + 1 - hour_branch) % 12
        return life_palace_branch

    # -----------------------------------------------------------------------
    # Step 2: 命宫 stem via 五虎遁 and 五行局
    # -----------------------------------------------------------------------

    def _get_life_palace_stem(self, dt: datetime.datetime, life_palace_branch: int) -> int:
        """Determine 命宫 stem using 五虎遁.

        The year stem determines the stem of the 寅(2) month.
        Then count forward from 寅 to 命宫 branch to get the stem.
        """
        pillars = gregorian_to_pillars(dt)
        year_stem = pillars[0][0]

        # 五虎遁: stem of 寅 month given year stem
        yin_stem = FIVE_TIGERS[year_stem]

        # Count forward from 寅(2) to 命宫 branch
        offset = (life_palace_branch - 2) % 12
        life_palace_stem = (yin_stem + offset) % 10
        return life_palace_stem

    def _get_bureau(self, stem: int, branch: int) -> int:
        """Determine 五行局 number from 命宫 stem + branch via 纳音五行.

        Returns one of: 2 (Water), 3 (Wood), 4 (Metal), 5 (Earth), 6 (Fire).
        """
        nayin_idx = _get_nayin_index(stem, branch)
        nayin_element = NAYIN_ELEMENTS[nayin_idx]
        bureau_map = {
            "water": 2,
            "wood":  3,
            "metal": 4,
            "earth": 5,
            "fire":  6,
        }
        return bureau_map[nayin_element]

    # -----------------------------------------------------------------------
    # Step 4: Star placement
    # -----------------------------------------------------------------------

    def _place_stars(self, ziwei_palace: int, life_palace_branch: int) -> dict:
        """Place all 14 major stars.

        紫微系: placed relative to 紫微's palace using fixed offsets.
        天府系: 天府 is placed at (命宫_branch * 2 - 紫微_palace) % 12,
               remaining 天府系 stars are placed relative to 天府.

        Args:
            ziwei_palace: 紫微's palace branch index (0-11)
            life_palace_branch: 命宫 branch index (0-11)

        Returns:
            {star_name: palace_branch_index (0-11)}
        """
        placement = {}

        # Place 紫微系 stars
        for star, offset in ZIWEI_SERIES_OFFSETS.items():
            palace = (ziwei_palace + offset) % 12
            placement[star] = palace

        # Determine 天府's palace
        # Standard formula: 天府 is the "mirror" of 紫微 across 命宫
        # 天府_palace = (命宫_branch * 2 - 紫微_palace) % 12
        tianfu_palace = (life_palace_branch * 2 - ziwei_palace) % 12

        # Place 天府系 stars
        for star, offset in TIANFU_SERIES_OFFSETS.items():
            palace = (tianfu_palace + offset) % 12
            placement[star] = palace

        return placement

    # -----------------------------------------------------------------------
    # Step 5: Palace scoring
    # -----------------------------------------------------------------------

    def _score_palaces(
        self,
        star_placement: dict,
        life_palace_branch: int,
    ) -> dict:
        """Compute sigmoid-normalized score for each of the 12 palaces.

        For each palace, sum contributions from all stars in that palace:
          raw = Σ star_base_value × brightness_weight
          score = sigmoid(raw)

        Star base values (from spec):
          auspicious: 庙旺=+2, 利=+1, 陷=0
          inauspicious: 庙旺=-2, 利=-1, 陷=0
          neutral: 庙旺=+1, 利=+0.5, 陷=0

        Brightness weights: 庙=1.0, 旺=0.75, 利=0.5, 陷=0.25

        Args:
            star_placement: {star_name: palace_branch_index}
            life_palace_branch: 命宫 branch index (used for palace labeling only)

        Returns:
            {palace_name: score in [0,1]}
        """
        # Accumulate raw scores per palace branch index
        raw_scores = {i: 0.0 for i in range(12)}

        for star, palace_branch in star_placement.items():
            nature = STAR_NATURE[star]
            brightness = STAR_BRIGHTNESS[star][palace_branch]

            # Compute base value based on nature and brightness level
            if nature == "auspicious":
                if brightness >= 1.0:      # 庙
                    base_value = +2.0
                elif brightness >= 0.75:   # 旺
                    base_value = +2.0
                elif brightness >= 0.5:    # 利
                    base_value = +1.0
                else:                      # 陷
                    base_value = 0.0
            elif nature == "inauspicious":
                if brightness >= 1.0:      # 庙
                    base_value = -2.0
                elif brightness >= 0.75:   # 旺
                    base_value = -2.0
                elif brightness >= 0.5:    # 利
                    base_value = -1.0
                else:                      # 陷
                    base_value = 0.0
            else:  # neutral (贪狼)
                if brightness >= 1.0:
                    base_value = +1.0
                elif brightness >= 0.75:
                    base_value = +1.0
                elif brightness >= 0.5:
                    base_value = +0.5
                else:
                    base_value = 0.0

            raw_scores[palace_branch] += base_value * brightness

        # Convert raw scores to palace names
        palace_scores = {}
        for offset, palace_name in enumerate(PALACES):
            absolute_branch = (life_palace_branch + offset) % 12
            palace_scores[palace_name] = _sigmoid(raw_scores[absolute_branch])

        return palace_scores

    # -----------------------------------------------------------------------
    # Step 6: Domain scores
    # -----------------------------------------------------------------------

    def _compute_domain_scores(
        self,
        palace_scores: dict,
        life_palace_branch: int,
    ) -> dict:
        """Map palace scores to 5 life domain scores.

        Domains map to palace offsets from 命宫:
          career       → 事业宫 (+8)
          wealth       → 财帛宫 (+4)
          relationships → 夫妻宫 (+2)
          health       → 疾厄宫 (+5)
          overall      → 命宫 (+0)
        """
        domain_scores = {}
        for domain, offset in DOMAIN_PALACE_OFFSET.items():
            palace_name = PALACES[offset]
            domain_scores[domain] = palace_scores[palace_name]
        return domain_scores
