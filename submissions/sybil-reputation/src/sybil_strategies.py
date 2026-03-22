"""Sybil attack strategies for reputation manipulation.

Each strategy function returns a list of (rater_id, ratee_id, rating) tuples
that the Sybil agents inject each round.
"""

from __future__ import annotations

from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import Agent


def ballot_stuffing(
    sybil_agents: List[Agent],
    honest_agents: List[Agent],
    rng,
) -> List[Tuple[int, int, float]]:
    """Sybil agents rate each other highly to inflate mutual reputations.

    Each Sybil rates all other Sybils with 0.95-1.0.
    """
    ratings = []
    for s1 in sybil_agents:
        for s2 in sybil_agents:
            if s1.agent_id != s2.agent_id:
                r = float(rng.uniform(0.95, 1.0))
                ratings.append((s1.agent_id, s2.agent_id, r))
    return ratings


def bad_mouthing(
    sybil_agents: List[Agent],
    honest_agents: List[Agent],
    rng,
) -> List[Tuple[int, int, float]]:
    """Sybil agents give very low ratings to top honest agents.

    Targets the top-3 honest agents (by true quality) with 0.0-0.1 ratings
    to suppress their reputations.
    """
    sorted_honest = sorted(honest_agents, key=lambda a: a.true_quality, reverse=True)
    targets = sorted_honest[:3]

    ratings = []
    for s in sybil_agents:
        for t in targets:
            r = float(rng.uniform(0.0, 0.1))
            ratings.append((s.agent_id, t.agent_id, r))
        # Also inflate each other
        for s2 in sybil_agents:
            if s.agent_id != s2.agent_id:
                r = float(rng.uniform(0.90, 1.0))
                ratings.append((s.agent_id, s2.agent_id, r))
    return ratings


def whitewashing(
    sybil_agents: List[Agent],
    honest_agents: List[Agent],
    rng,
) -> List[Tuple[int, int, float]]:
    """Sybil agents periodically reset by creating fresh identities.

    They act honestly for a while (rating ~ true quality) then switch
    to inflating each other. The "whitewashing" is modeled by resetting
    account_age to 0 on Sybil agents that have accumulated bad reputation.

    In practice this function gives mixed ratings: honest-ish to non-sybils,
    inflated to fellow Sybils, to simulate the transition period.
    """
    ratings = []
    for s in sybil_agents:
        # Rate honest agents somewhat honestly to build credibility
        for h in honest_agents[:5]:
            r = float(rng.uniform(0.3, 0.7))
            ratings.append((s.agent_id, h.agent_id, r))
        # But still inflate each other
        for s2 in sybil_agents:
            if s.agent_id != s2.agent_id:
                r = float(rng.uniform(0.85, 1.0))
                ratings.append((s.agent_id, s2.agent_id, r))
    return ratings


STRATEGIES = {
    "ballot_stuffing": ballot_stuffing,
    "bad_mouthing": bad_mouthing,
    "whitewashing": whitewashing,
}
