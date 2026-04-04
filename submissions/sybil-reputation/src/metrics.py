"""Metrics for evaluating reputation system resilience to Sybil attacks.

Four primary metrics, each in [0, 1] or [-1, 1]:

1. reputation_accuracy  -- Spearman correlation between reputation scores and
                           true quality for honest agents only.
2. sybil_detection_rate -- Fraction of Sybil agents whose reputation is below
                           the median honest reputation (lower = detected).
3. honest_welfare       -- Mean reputation of honest agents (should be high
                           in a fair system).
4. market_efficiency    -- 1 - mean absolute error between reputation rank
                           and true-quality rank for honest agents, normalized.
"""

from __future__ import annotations

import math
import statistics
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import Agent


def _mean(values: List[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _rankdata(values: List[float]) -> List[float]:
    indexed = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0 for _ in values]
    pos = 0
    while pos < len(indexed):
        end = pos + 1
        while end < len(indexed) and indexed[end][0] == indexed[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for _, idx in indexed[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    return ranks


def _pearson(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0
    x_mean = _mean(x)
    y_mean = _mean(y)
    num = 0.0
    x_var = 0.0
    y_var = 0.0
    for i in range(len(x)):
        dx = x[i] - x_mean
        dy = y[i] - y_mean
        num += dx * dy
        x_var += dx * dx
        y_var += dy * dy
    if x_var <= 0 or y_var <= 0:
        return 0.0
    return num / math.sqrt(x_var * y_var)


def _spearmanr(x: List[float], y: List[float]) -> float:
    return _pearson(_rankdata(x), _rankdata(y))


def _sign(value: float) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def _kendalltau(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0

    for i in range(len(x) - 1):
        for j in range(i + 1, len(x)):
            dx = _sign(x[i] - x[j])
            dy = _sign(y[i] - y[j])
            if dx == 0 and dy == 0:
                ties_x += 1
                ties_y += 1
            elif dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1

    denom = math.sqrt(
        (concordant + discordant + ties_x) * (concordant + discordant + ties_y)
    )
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom


def reputation_accuracy(
    honest_agents: List[Agent], scores: Dict[int, float]
) -> float:
    """Spearman rank correlation between reputation and true quality.

    Returns correlation in [-1, 1]. Higher is better.
    Returns 0.0 if fewer than 3 honest agents have scores.
    """
    qualities = []
    reps = []
    for a in honest_agents:
        if a.agent_id in scores:
            qualities.append(a.true_quality)
            reps.append(scores[a.agent_id])
    if len(qualities) < 3:
        return 0.0
    corr = _spearmanr(qualities, reps)
    if math.isnan(corr):
        return 0.0
    return float(corr)


def sybil_detection_rate(
    honest_agents: List[Agent],
    sybil_agents: List[Agent],
    scores: Dict[int, float],
) -> float:
    """Fraction of Sybil agents with reputation below median honest reputation.

    Higher means the system correctly keeps Sybils below honest agents.
    Returns 0.0 if no Sybil agents.
    """
    if len(sybil_agents) == 0:
        return 1.0  # No Sybils to detect -- perfect by definition

    honest_scores = [scores.get(a.agent_id, 0.5) for a in honest_agents]
    median_honest = _median(honest_scores)

    detected = sum(
        1 for a in sybil_agents if scores.get(a.agent_id, 0.5) < median_honest
    )
    return detected / len(sybil_agents)


def honest_welfare(
    honest_agents: List[Agent], scores: Dict[int, float]
) -> float:
    """Mean reputation score of honest agents. Higher is better."""
    if len(honest_agents) == 0:
        return 0.0
    vals = [scores.get(a.agent_id, 0.5) for a in honest_agents]
    return _mean(vals)


def market_efficiency(
    honest_agents: List[Agent], scores: Dict[int, float]
) -> float:
    """How well reputation ranking matches true-quality ranking.

    Computed as 1 - normalized Kendall tau distance.
    Returns value in [0, 1] where 1 = perfect ranking.
    Returns 0.5 if fewer than 3 agents.
    """
    qualities = []
    reps = []
    for a in honest_agents:
        if a.agent_id in scores:
            qualities.append(a.true_quality)
            reps.append(scores[a.agent_id])
    if len(qualities) < 3:
        return 0.5
    tau = _kendalltau(qualities, reps)
    if math.isnan(tau):
        return 0.5
    # Convert from [-1, 1] to [0, 1]
    return float((tau + 1) / 2)


def compute_all_metrics(
    honest_agents: List[Agent],
    sybil_agents: List[Agent],
    scores: Dict[int, float],
) -> Dict[str, float]:
    """Compute all four metrics and return as a dict."""
    return {
        "reputation_accuracy": reputation_accuracy(honest_agents, scores),
        "sybil_detection_rate": sybil_detection_rate(
            honest_agents, sybil_agents, scores
        ),
        "honest_welfare": honest_welfare(honest_agents, scores),
        "market_efficiency": market_efficiency(honest_agents, scores),
    }
