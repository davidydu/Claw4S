"""Reputation algorithms for multi-agent networks.

Implements four algorithms with increasing Sybil resilience:
1. SimpleAverage  -- mean of all ratings received
2. WeightedByHistory -- ratings weighted by rater account age
3. PageRankTrust -- trust flows through transaction graph
4. EigenTrust -- Kamvar et al. 2003 iterative trust algorithm
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .agents import Agent

# ---------------------------------------------------------------------------
# Rating ledger: list of (rater_id, ratee_id, rating_value, round_number)
# ---------------------------------------------------------------------------
Rating = Tuple[int, int, float, int]


def simple_average(agents: List[Agent], ledger: List[Rating]) -> Dict[int, float]:
    """Compute reputation as mean of all ratings received.

    This is the baseline -- no Sybil defense at all.
    """
    sums: Dict[int, float] = {}
    counts: Dict[int, int] = {}
    for _rater, ratee, value, _rnd in ledger:
        sums[ratee] = sums.get(ratee, 0.0) + value
        counts[ratee] = counts.get(ratee, 0) + 1

    scores = {}
    for a in agents:
        if counts.get(a.agent_id, 0) > 0:
            scores[a.agent_id] = sums[a.agent_id] / counts[a.agent_id]
        else:
            scores[a.agent_id] = 0.5  # prior
    return scores


def weighted_by_history(
    agents: List[Agent], ledger: List[Rating], current_round: int
) -> Dict[int, float]:
    """Ratings weighted by rater's account age (older = more trusted).

    Weight = log2(2 + account_age) where account_age = current_round - first_seen.
    """
    first_seen: Dict[int, int] = {}
    for rater, _ratee, _val, rnd in ledger:
        if rater not in first_seen:
            first_seen[rater] = rnd

    weighted_sums: Dict[int, float] = {}
    weight_totals: Dict[int, float] = {}

    for rater, ratee, value, _rnd in ledger:
        age = current_round - first_seen.get(rater, current_round)
        w = float(np.log2(2 + age))
        weighted_sums[ratee] = weighted_sums.get(ratee, 0.0) + w * value
        weight_totals[ratee] = weight_totals.get(ratee, 0.0) + w

    scores = {}
    for a in agents:
        wt = weight_totals.get(a.agent_id, 0.0)
        if wt > 0:
            scores[a.agent_id] = weighted_sums[a.agent_id] / wt
        else:
            scores[a.agent_id] = 0.5
    return scores


def pagerank_trust(
    agents: List[Agent], ledger: List[Rating],
    damping: float = 0.85, iterations: int = 30,
) -> Dict[int, float]:
    """PageRank-style trust propagation through the transaction graph.

    Build a directed graph from transactions (rater -> ratee with positive
    rating creates an edge). Then run PageRank and normalize to [0, 1].
    """
    id_list = [a.agent_id for a in agents]
    n = len(id_list)
    if n == 0:
        return {}
    id_to_idx = {aid: i for i, aid in enumerate(id_list)}

    # Build adjacency: edge weight = mean positive rating from i to j
    edge_sums = np.zeros((n, n), dtype=np.float64)
    edge_counts = np.zeros((n, n), dtype=np.float64)

    for rater, ratee, value, _rnd in ledger:
        if rater in id_to_idx and ratee in id_to_idx and value > 0.5:
            i, j = id_to_idx[rater], id_to_idx[ratee]
            edge_sums[i, j] += value
            edge_counts[i, j] += 1

    # Normalize rows to create transition matrix
    adj = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        total = edge_counts[i].sum()
        if total > 0:
            adj[i] = edge_sums[i] / (edge_sums[i].sum() + 1e-12)

    # PageRank iteration
    rank = np.ones(n, dtype=np.float64) / n
    teleport = np.ones(n, dtype=np.float64) / n
    for _ in range(iterations):
        rank = (1 - damping) * teleport + damping * adj.T @ rank
        rank_sum = rank.sum()
        if rank_sum > 0:
            rank /= rank_sum

    # Normalize to [0, 1]
    rmin, rmax = rank.min(), rank.max()
    if rmax > rmin:
        rank = (rank - rmin) / (rmax - rmin)
    else:
        rank = np.full(n, 0.5)

    return {id_list[i]: float(rank[i]) for i in range(n)}


def eigentrust(
    agents: List[Agent], ledger: List[Rating],
    iterations: int = 30, alpha: float = 0.1,
) -> Dict[int, float]:
    """EigenTrust algorithm (Kamvar et al. 2003).

    Computes trust from local trust values (satisfaction scores)
    using iterative left-multiplication by a normalized trust matrix,
    blended with a prior trust vector.
    """
    id_list = [a.agent_id for a in agents]
    n = len(id_list)
    if n == 0:
        return {}
    id_to_idx = {aid: i for i, aid in enumerate(id_list)}

    # Build local trust: s_{ij} = sum of positive ratings from i about j
    # minus sum of negative ratings (below 0.5)
    s = np.zeros((n, n), dtype=np.float64)
    for rater, ratee, value, _rnd in ledger:
        if rater in id_to_idx and ratee in id_to_idx:
            i, j = id_to_idx[rater], id_to_idx[ratee]
            s[i, j] += value - 0.5  # Centered: positive = good

    # Clip negatives, normalize rows
    s = np.maximum(s, 0)
    c = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        row_sum = s[i].sum()
        if row_sum > 0:
            c[i] = s[i] / row_sum
        else:
            c[i] = 1.0 / n  # uniform prior

    # Pre-trusted peers: uniform distribution
    p = np.ones(n, dtype=np.float64) / n

    # Iterate: t^{k+1} = (1-alpha) * C^T * t^k + alpha * p
    t = p.copy()
    for _ in range(iterations):
        t = (1 - alpha) * c.T @ t + alpha * p
        t_sum = t.sum()
        if t_sum > 0:
            t /= t_sum

    # Normalize to [0, 1]
    tmin, tmax = t.min(), t.max()
    if tmax > tmin:
        t = (t - tmin) / (tmax - tmin)
    else:
        t = np.full(n, 0.5)

    return {id_list[i]: float(t[i]) for i in range(n)}


ALGORITHMS = {
    "simple_average": simple_average,
    "weighted_history": weighted_by_history,
    "pagerank_trust": pagerank_trust,
    "eigentrust": eigentrust,
}
