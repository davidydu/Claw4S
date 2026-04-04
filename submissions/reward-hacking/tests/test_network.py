"""Tests for network topology generators."""

import numpy as np
import pytest

from src.network import build_adjacency, neighbor_list


class TestGridAdjacency:
    def test_shape(self):
        adj = build_adjacency(9, "grid", np.random.default_rng(0))
        assert adj.shape == (9, 9)

    def test_symmetric(self):
        adj = build_adjacency(10, "grid", np.random.default_rng(0))
        assert np.array_equal(adj, adj.T)

    def test_no_self_loops(self):
        adj = build_adjacency(10, "grid", np.random.default_rng(0))
        assert not np.any(np.diag(adj))

    def test_corner_has_two_neighbors(self):
        """On a 3x3 grid, agent 0 (top-left corner) has exactly 2 neighbors."""
        adj = build_adjacency(9, "grid", np.random.default_rng(0))
        assert adj[0].sum() == 2

    def test_center_has_four_neighbors(self):
        """On a 3x3 grid, agent 4 (center) has exactly 4 neighbors."""
        adj = build_adjacency(9, "grid", np.random.default_rng(0))
        assert adj[4].sum() == 4


class TestRandomAdjacency:
    def test_symmetric(self):
        adj = build_adjacency(10, "random", np.random.default_rng(42))
        assert np.array_equal(adj, adj.T)

    def test_no_self_loops(self):
        adj = build_adjacency(10, "random", np.random.default_rng(42))
        assert not np.any(np.diag(adj))

    def test_has_edges(self):
        adj = build_adjacency(10, "random", np.random.default_rng(42))
        assert adj.sum() > 0

    def test_deterministic(self):
        adj1 = build_adjacency(10, "random", np.random.default_rng(42))
        adj2 = build_adjacency(10, "random", np.random.default_rng(42))
        assert np.array_equal(adj1, adj2)


class TestStarAdjacency:
    def test_hub_connected_to_all(self):
        adj = build_adjacency(5, "star", np.random.default_rng(0))
        assert adj[0].sum() == 4

    def test_leaf_connected_to_hub_only(self):
        adj = build_adjacency(5, "star", np.random.default_rng(0))
        for i in range(1, 5):
            assert adj[i].sum() == 1
            assert adj[i, 0] is np.True_


class TestNeighborList:
    def test_matches_adjacency(self):
        adj = build_adjacency(9, "grid", np.random.default_rng(0))
        nbrs = neighbor_list(adj)
        for i in range(9):
            expected = set(np.where(adj[i])[0])
            assert set(nbrs[i]) == expected


class TestInvalidTopology:
    def test_raises(self):
        with pytest.raises(ValueError, match="Unknown topology"):
            build_adjacency(5, "invalid", np.random.default_rng(0))
