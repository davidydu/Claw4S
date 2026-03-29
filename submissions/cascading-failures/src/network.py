"""Network topology generators for multi-agent cascade simulations.

Generates adjacency lists for six topologies:
  chain, ring, star, erdos_renyi, scale_free, fully_connected.

Each generator returns a dict mapping node_id -> list[neighbor_ids] (directed
edges point from dependency to dependent, i.e. neighbor outputs feed into
node's input).
"""

from __future__ import annotations

import random
from typing import Dict, List


AdjList = Dict[int, List[int]]


def _undirected_to_adj(edges: list[tuple[int, int]], n: int) -> AdjList:
    """Convert edge list to symmetric adjacency list."""
    adj: AdjList = {i: [] for i in range(n)}
    for u, v in edges:
        if v not in adj[u]:
            adj[u].append(v)
        if u not in adj[v]:
            adj[v].append(u)
    # Sort for determinism
    for k in adj:
        adj[k].sort()
    return adj


def chain(n: int) -> AdjList:
    """Linear chain: node i depends on node i-1."""
    adj: AdjList = {i: [] for i in range(n)}
    for i in range(1, n):
        adj[i].append(i - 1)
        adj[i - 1].append(i)
    for k in adj:
        adj[k].sort()
    return adj


def ring(n: int) -> AdjList:
    """Ring: chain plus edge from node 0 to node n-1."""
    adj = chain(n)
    if n > 2:
        adj[0].append(n - 1)
        adj[n - 1].append(0)
    for k in adj:
        adj[k].sort()
    return adj


def star(n: int) -> AdjList:
    """Star: node 0 is hub, all others connect only to hub."""
    adj: AdjList = {i: [] for i in range(n)}
    for i in range(1, n):
        adj[0].append(i)
        adj[i].append(0)
    for k in adj:
        adj[k].sort()
    return adj


def erdos_renyi(n: int, p: float = 0.2, rng: random.Random | None = None) -> AdjList:
    """Erdos-Renyi random graph G(n,p)."""
    if rng is None:
        rng = random.Random(42)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p:
                edges.append((i, j))
    return _undirected_to_adj(edges, n)


def scale_free(n: int, m: int = 2, rng: random.Random | None = None) -> AdjList:
    """Barabasi-Albert preferential attachment (scale-free) graph.

    Starts with a clique of m+1 nodes, then adds n-(m+1) nodes each
    attaching to m existing nodes with probability proportional to degree.
    """
    if rng is None:
        rng = random.Random(42)
    if n <= m + 1:
        # Return fully connected for tiny n
        return fully_connected(n)
    # Start with clique of m+1
    edges = []
    for i in range(m + 1):
        for j in range(i + 1, m + 1):
            edges.append((i, j))
    degree = [m] * (m + 1)  # each node in clique has degree m
    total_degree = (m + 1) * m
    for new_node in range(m + 1, n):
        # Choose m distinct targets by preferential attachment
        targets = set()
        while len(targets) < m:
            r = rng.random() * total_degree
            cumulative = 0
            for node_id in range(new_node):
                cumulative += degree[node_id]
                if cumulative > r:
                    targets.add(node_id)
                    break
        for t in targets:
            edges.append((new_node, t))
        degree.append(len(targets))
        for t in targets:
            degree[t] += 1
        total_degree += 2 * len(targets)
    return _undirected_to_adj(edges, n)


def fully_connected(n: int) -> AdjList:
    """Fully connected (complete) graph."""
    adj: AdjList = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j:
                adj[i].append(j)
    for k in adj:
        adj[k].sort()
    return adj


TOPOLOGIES = {
    "chain": chain,
    "ring": ring,
    "star": star,
    "erdos_renyi": erdos_renyi,
    "scale_free": scale_free,
    "fully_connected": fully_connected,
}


def build_network(name: str, n: int, seed: int = 42) -> AdjList:
    """Build a network by name with given size and seed."""
    rng = random.Random(seed)
    if name == "erdos_renyi":
        return erdos_renyi(n, p=0.2, rng=rng)
    elif name == "scale_free":
        return scale_free(n, m=2, rng=rng)
    else:
        return TOPOLOGIES[name](n)


def hub_node(adj: AdjList) -> int:
    """Return the node with highest degree (hub)."""
    return max_degree_nodes(adj)[0]


def max_degree_nodes(adj: AdjList) -> List[int]:
    """Return all nodes with the maximum degree, sorted ascending."""
    if not adj:
        raise ValueError("adjacency list must include at least one node")
    max_degree = max(len(neighbors) for neighbors in adj.values())
    return sorted([node for node, neighbors in adj.items() if len(neighbors) == max_degree])
