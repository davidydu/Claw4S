"""Tests for symmetry metrics."""

import torch
from src.metrics import pairwise_cosine_similarity, symmetry_metric


def test_identical_rows_symmetry_one():
    """Identical rows should have symmetry metric ~1.0."""
    w = torch.tensor([[1.0, 2.0, 3.0]] * 4)
    assert abs(symmetry_metric(w) - 1.0) < 1e-5


def test_orthogonal_rows_symmetry_zero():
    """Orthogonal rows should have symmetry metric ~0.0."""
    w = torch.eye(4)
    assert abs(symmetry_metric(w) - 0.0) < 1e-5


def test_cosine_sim_diagonal_one():
    """Diagonal of cosine similarity matrix should be 1.0."""
    w = torch.randn(5, 10)
    sim = pairwise_cosine_similarity(w)
    assert torch.allclose(sim.diag(), torch.ones(5), atol=1e-5)


def test_cosine_sim_symmetric():
    """Cosine similarity matrix should be symmetric."""
    w = torch.randn(5, 10)
    sim = pairwise_cosine_similarity(w)
    assert torch.allclose(sim, sim.t(), atol=1e-5)


def test_symmetry_metric_range():
    """Symmetry metric should be in [-1, 1]."""
    w = torch.randn(8, 16)
    m = symmetry_metric(w)
    assert -1.0 <= m <= 1.0 + 1e-5, f"Symmetry metric {m} out of range"


def test_single_neuron():
    """Single neuron should return symmetry 1.0 (degenerate case)."""
    w = torch.randn(1, 10)
    assert symmetry_metric(w) == 1.0
