# src/ziwei_tables.py
"""Zi Wei Dou Shu lookup tables: star placement, offsets, 庙旺利陷.

Palace indices correspond to the 12 earthly branches in order:
  0=子, 1=丑, 2=寅, 3=卯, 4=辰, 5=巳, 6=午, 7=未, 8=申, 9=酉, 10=戌, 11=亥

The 12 palaces (宫位) are arranged in the chart starting from 命宫.
Palace positions are absolute branch indices (0-11), not relative offsets.
"""

# ---------------------------------------------------------------------------
# 紫微星 (Zi Wei) placement table
# ---------------------------------------------------------------------------
# Keys: (day_number, bureau_number) → palace_index (0-11)
# Bureau numbers: 水=2, 木=3, 金=4, 土=5, 火=6
# Day numbers: 1-30 (lunar day)
#
# Algorithm (standard Zi Wei reference):
#   quotient = (day - 1) // bureau
#   remainder = (day - 1) % bureau
#   if remainder == 0: palace = (quotient + 1) % 12
#   else:              palace = (quotient + bureau + 1) % 12
# ---------------------------------------------------------------------------

def _compute_ziwei_palace(day, bureau):
    """Standard 紫微星 placement algorithm."""
    quotient = (day - 1) // bureau
    remainder = (day - 1) % bureau
    if remainder == 0:
        palace = (quotient + 1) % 12
    else:
        palace = (quotient + bureau + 1) % 12
    return palace


ZIWEI_PLACEMENT = {
    (day, bureau): _compute_ziwei_palace(day, bureau)
    for bureau in (2, 3, 4, 5, 6)
    for day in range(1, 31)
}

# ---------------------------------------------------------------------------
# Star offsets relative to 紫微 (紫微系 — Zi Wei series)
# ---------------------------------------------------------------------------
# Format: {star_name: offset_from_ziwei}
# Positive = counter-clockwise (incrementing branch index), wraps mod 12
# These are the standard offsets from orthodox Zi Wei references.
# The 紫微系 stars sit in palaces relative to 紫微's palace.
ZIWEI_SERIES_OFFSETS = {
    "紫微": 0,
    "天机": -1,   # 1 palace counter-clockwise (i.e. branch - 1)
    "太阳": -3,   # 3 palaces counter-clockwise
    "武曲": -4,
    "天同": -5,
    "廉贞": -8,   # equivalent to +4 mod 12
}

# ---------------------------------------------------------------------------
# 天府系 star offsets (relative to 天府's palace)
# ---------------------------------------------------------------------------
# 天府's palace is derived at runtime from: (命宫_branch * 2 - 紫微_palace) % 12
# The remaining 天府系 stars are placed relative to 天府.
TIANFU_SERIES_OFFSETS = {
    "天府": 0,
    "太阴": 1,
    "贪狼": 2,
    "巨门": 3,
    "天相": 4,
    "天梁": 5,
    "七杀": 6,
    "破军": 10,   # breaks the consecutive pattern — standard placement
}

# All 14 major stars for validation
ALL_MAJOR_STARS = list(ZIWEI_SERIES_OFFSETS.keys()) + list(TIANFU_SERIES_OFFSETS.keys())
assert len(ALL_MAJOR_STARS) == 14, f"Expected 14 major stars, got {len(ALL_MAJOR_STARS)}"

# ---------------------------------------------------------------------------
# 14 major stars classified as 吉/凶/neutral
# ---------------------------------------------------------------------------
# Nature determines sign of base_value contribution in scoring:
#   auspicious  → positive contribution
#   inauspicious → negative contribution
#   neutral     → context-dependent (treated as slightly positive here)
STAR_NATURE = {
    "紫微": "auspicious",
    "天机": "auspicious",
    "太阳": "auspicious",
    "武曲": "auspicious",
    "天同": "auspicious",
    "廉贞": "inauspicious",
    "天府": "auspicious",
    "太阴": "auspicious",
    "贪狼": "neutral",
    "巨门": "inauspicious",
    "天相": "auspicious",
    "天梁": "auspicious",
    "七杀": "inauspicious",
    "破军": "inauspicious",
}

# ---------------------------------------------------------------------------
# 庙旺利陷 brightness table
# ---------------------------------------------------------------------------
# Format: {star_name: [brightness_level_0, ..., brightness_level_11]}
# Index = palace (branch) index: 0=子, 1=丑, 2=寅, ..., 11=亥
# Levels: 庙=1.0, 旺=0.75, 利=0.5, 陷=0.25
#
# Based on standard Zi Wei Dou Shu 庙旺利陷 references.
# Yang branches (子午卯酉 = 0,6,3,9) are strong for yang stars.
# Yin branches (丑未辰戌 = 1,7,4,10) are strong for earth/yin stars.
# Fire/Wood stars tend to be strong in spring/summer branches.
# ---------------------------------------------------------------------------

# Helper: build a 12-entry brightness list by specifying levels for groups
def _brightness(miao, wang, li, xian):
    """Create a 12-entry brightness array.

    Args:
        miao: list of palace indices at 庙 level (1.0)
        wang: list of palace indices at 旺 level (0.75)
        li:   list of palace indices at 利 level (0.5)
        xian: list of palace indices at 陷 level (0.25)
    """
    result = [0.5] * 12  # default to 利
    for i in miao:
        result[i] = 1.0
    for i in wang:
        result[i] = 0.75
    for i in li:
        result[i] = 0.5
    for i in xian:
        result[i] = 0.25
    return result


# Branch group references (for readability):
# 子=0, 丑=1, 寅=2, 卯=3, 辰=4, 巳=5, 午=6, 未=7, 申=8, 酉=9, 戌=10, 亥=11

STAR_BRIGHTNESS = {
    # 紫微: 庙 in 子午, 旺 in 寅申巳亥, 利 in 辰戌, 陷 in 丑未卯酉
    "紫微": _brightness(
        miao=[0, 6],
        wang=[2, 8, 5, 11],
        li=[4, 10],
        xian=[1, 7, 3, 9],
    ),

    # 天机: 庙 in 卯酉 (Wood star, strong in wood/metal branches),
    #        旺 in 子午, 利 in 寅亥, 陷 in 辰戌丑未
    "天机": _brightness(
        miao=[3, 9],
        wang=[0, 6],
        li=[2, 11, 4, 10],
        xian=[1, 7, 5, 8],
    ),

    # 太阳: 庙 in 寅卯辰 (rising/midday, Fire/yang energy)
    #        旺 in 午, 利 in 巳未, 陷 in 申酉戌亥子丑 (setting/night)
    "太阳": _brightness(
        miao=[2, 3, 4],
        wang=[6],
        li=[5, 7],
        xian=[8, 9, 10, 11, 0, 1],
    ),

    # 武曲: 庙 in 丑未 (Metal earth), 旺 in 辰戌, 利 in 子午, 陷 in 寅卯巳申酉亥
    "武曲": _brightness(
        miao=[1, 7],
        wang=[4, 10],
        li=[0, 6],
        xian=[2, 3, 5, 8, 9, 11],
    ),

    # 天同: 庙 in 子亥 (Water branches), 旺 in 午未, 利 in 丑寅, 陷 in 卯辰巳申酉戌
    "天同": _brightness(
        miao=[0, 11],
        wang=[6, 7],
        li=[1, 2],
        xian=[3, 4, 5, 8, 9, 10],
    ),

    # 廉贞: 庙 in 寅午 (inauspicious but strong in Fire/Wood),
    #        旺 in 巳卯, 利 in 子申, 陷 in 丑辰未戌酉亥
    "廉贞": _brightness(
        miao=[2, 6],
        wang=[5, 3],
        li=[0, 8],
        xian=[1, 4, 7, 10, 9, 11],
    ),

    # 天府: 庙 in 寅申 (yang metal/wood), 旺 in 子午巳亥, 利 in 辰戌卯酉, 陷 in 丑未
    "天府": _brightness(
        miao=[2, 8],
        wang=[0, 6, 5, 11],
        li=[4, 10, 3, 9],
        xian=[1, 7],
    ),

    # 太阴: 庙 in 子亥丑 (Water yin), 旺 in 寅卯, 利 in 辰, 陷 in 巳午未申酉戌
    "太阴": _brightness(
        miao=[0, 11, 1],
        wang=[2, 3],
        li=[4],
        xian=[5, 6, 7, 8, 9, 10],
    ),

    # 贪狼: 庙 in 寅卯 (Wood), 旺 in 亥子, 利 in 辰戌, 陷 in 丑未午巳申酉
    "贪狼": _brightness(
        miao=[2, 3],
        wang=[11, 0],
        li=[4, 10],
        xian=[1, 7, 6, 5, 8, 9],
    ),

    # 巨门: 庙 in 子午 (strong axis), 旺 in 丑未, 利 in 寅申, 陷 in 卯辰巳酉戌亥
    "巨门": _brightness(
        miao=[0, 6],
        wang=[1, 7],
        li=[2, 8],
        xian=[3, 4, 5, 9, 10, 11],
    ),

    # 天相: 庙 in 寅午 (Water+Fire), 旺 in 子辰, 利 in 申亥, 陷 in 丑卯巳未酉戌
    "天相": _brightness(
        miao=[2, 6],
        wang=[0, 4],
        li=[8, 11],
        xian=[1, 3, 5, 7, 9, 10],
    ),

    # 天梁: 庙 in 午戌 (auspicious, strong in Fire earth), 旺 in 寅申, 利 in 子辰, 陷 in 丑卯巳未酉亥
    "天梁": _brightness(
        miao=[6, 10],
        wang=[2, 8],
        li=[0, 4],
        xian=[1, 3, 5, 7, 9, 11],
    ),

    # 七杀: 庙 in 子午卯酉 (strong on cardinal branches), 旺 in 寅申, 利 in 辰戌, 陷 in 丑巳未亥
    "七杀": _brightness(
        miao=[0, 6, 3, 9],
        wang=[2, 8],
        li=[4, 10],
        xian=[1, 5, 7, 11],
    ),

    # 破军: 庙 in 子午 (Water Fire axis), 旺 in 辰戌丑未, 利 in 寅申, 陷 in 卯巳酉亥
    "破军": _brightness(
        miao=[0, 6],
        wang=[4, 10, 1, 7],
        li=[2, 8],
        xian=[3, 5, 9, 11],
    ),
}

# Validate: every star has 12 brightness entries
for _star, _vals in STAR_BRIGHTNESS.items():
    assert len(_vals) == 12, f"{_star} has {len(_vals)} brightness entries, expected 12"

# ---------------------------------------------------------------------------
# 12 宫位 (palaces) — in order starting from 命宫 (index 0)
# ---------------------------------------------------------------------------
# The palace at position offset P from 命宫 is:
#   absolute_branch = (命宫_branch + P) % 12
PALACES = [
    "命宫",   # 0 — Life palace
    "兄弟",   # 1 — Siblings
    "夫妻",   # 2 — Spouse/Relationships
    "子女",   # 3 — Children
    "财帛",   # 4 — Wealth
    "疾厄",   # 5 — Health
    "迁移",   # 6 — Travel/Migration
    "交友",   # 7 — Friends
    "事业",   # 8 — Career
    "田宅",   # 9 — Property
    "福德",   # 10 — Blessings
    "父母",   # 11 — Parents
]

# Domain → palace offset from 命宫
DOMAIN_PALACE_OFFSET = {
    "career":        8,   # 事业宫
    "wealth":        4,   # 财帛宫
    "relationships": 2,   # 夫妻宫
    "health":        5,   # 疾厄宫
    "overall":       0,   # 命宫
}
