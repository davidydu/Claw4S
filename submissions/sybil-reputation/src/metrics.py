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

import numpy as np
from scipy import stats
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import Agent


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
    corr, _ = stats.spearmanr(qualities, reps)
    if np.isnan(corr):
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
    median_honest = float(np.median(honest_scores))

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
    return float(np.mean(vals))


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
    tau, _ = stats.kendalltau(qualities, reps)
    if np.isnan(tau):
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
