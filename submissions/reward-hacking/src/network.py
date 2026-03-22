"""Network topology generators for multi-agent systems.

Supported topologies:
  - grid: agents placed on a sqrt(N) x sqrt(N) grid with 4-connectivity
  - random: Erdos-Renyi graph with edge probability 0.3
  - star: one central hub connected to all others
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def build_adjacency(n: int, topology: str, rng: np.random.Generator) -> NDArray[np.bool_]:
    """Return an N x N boolean adjacency matrix (symmetric, no self-loops).

    Parameters
    ----------
    n : int
        Number of agents.
    topology : str
        One of "grid", "random", "star".
    rng : numpy.random.Generator
        Random number generator (used only for "random" topology).

    Returns
    -------
    adj : ndarray of shape (n, n), dtype bool
    """
    if topology == "grid":
        return _grid_adjacency(n)
    elif topology == "random":
        return _random_adjacency(n, rng, edge_prob=0.3)
    elif topology == "star":
        return _star_adjacency(n)
    else:
        raise ValueError(f"Unknown topology: {topology!r}")


def _grid_adjacency(n: int) -> NDArray[np.bool_]:
    side = int(np.ceil(np.sqrt(n)))
    adj = np.zeros((n, n), dtype=bool)
    for i in range(n):
        row_i, col_i = divmod(i, side)
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            row_j, col_j = row_i + di, col_i + dj
            j = row_j * side + col_j
            if 0 <= row_j < side and 0 <= col_j < side and 0 <= j < n:
                adj[i, j] = True
    return adj


def _random_adjacency(n: int, rng: np.random.Generator, edge_prob: float) -> NDArray[np.bool_]:
    upper = rng.random((n, n)) < edge_prob
    adj = np.triu(upper, k=1)
    adj = adj | adj.T
    return adj


def _star_adjacency(n: int) -> NDArray[np.bool_]:
    adj = np.zeros((n, n), dtype=bool)
    for i in range(1, n):
        adj[0, i] = True
        adj[i, 0] = True
    return adj


def neighbor_list(adj: NDArray[np.bool_]) -> list[list[int]]:
    """Convert adjacency matrix to list-of-neighbors for fast lookups."""
    n = adj.shape[0]
    return [list(np.where(adj[i])[0]) for i in range(n)]
