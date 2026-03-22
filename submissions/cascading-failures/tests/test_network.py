"""Tests for network topology generators."""

import random
from src.network import (
    chain, ring, star, erdos_renyi, scale_free, fully_connected,
    build_network, hub_node,
)


def test_chain_structure():
    adj = chain(5)
    assert len(adj) == 5
    # Node 0 connects to node 1 only
    assert adj[0] == [1]
    # Node 2 connects to 1 and 3
    assert adj[2] == [1, 3]
    # Node 4 (end) connects to 3 only
    assert adj[4] == [3]


def test_ring_closes_loop():
    adj = ring(5)
    assert 4 in adj[0]
    assert 0 in adj[4]


def test_star_hub():
    adj = star(10)
    # Hub (node 0) connects to all others
    assert len(adj[0]) == 9
    # Each spoke connects only to hub
    for i in range(1, 10):
        assert adj[i] == [0]


def test_erdos_renyi_deterministic():
    rng1 = random.Random(42)
    rng2 = random.Random(42)
    adj1 = erdos_renyi(10, p=0.3, rng=rng1)
    adj2 = erdos_renyi(10, p=0.3, rng=rng2)
    assert adj1 == adj2


def test_erdos_renyi_connectivity():
    adj = erdos_renyi(20, p=0.2, rng=random.Random(42))
    assert len(adj) == 20
    # With p=0.2 and n=20, most nodes should have some neighbors
    connected = sum(1 for v in adj.values() if len(v) > 0)
    assert connected >= 15  # Very likely with p=0.2


def test_scale_free_degree_distribution():
    adj = scale_free(20, m=2, rng=random.Random(42))
    assert len(adj) == 20
    degrees = [len(adj[i]) for i in range(20)]
    max_deg = max(degrees)
    min_deg = min(degrees)
    # Scale-free should have heterogeneous degrees
    assert max_deg > min_deg


def test_fully_connected():
    adj = fully_connected(5)
    for i in range(5):
        assert len(adj[i]) == 4
        assert i not in adj[i]


def test_build_network_all_topologies():
    names = ["chain", "ring", "star", "erdos_renyi", "scale_free", "fully_connected"]
    for name in names:
        adj = build_network(name, 10, seed=42)
        assert len(adj) == 10, f"{name} should have 10 nodes"


def test_hub_node_star():
    adj = star(10)
    assert hub_node(adj) == 0


def test_symmetry():
    """All topologies should produce symmetric adjacency lists."""
    for name in ["chain", "ring", "star", "erdos_renyi", "scale_free", "fully_connected"]:
        adj = build_network(name, 10, seed=42)
        for u in adj:
            for v in adj[u]:
                assert u in adj[v], f"{name}: edge {u}->{v} but not {v}->{u}"
