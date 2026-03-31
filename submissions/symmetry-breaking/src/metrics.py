"""Symmetry metrics for measuring how similar hidden neurons are."""

import torch
from typing import List


def pairwise_cosine_similarity(weight_matrix: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between rows of a weight matrix.

    Args:
        weight_matrix: Shape (num_neurons, input_dim).

    Returns:
        Symmetric matrix of shape (num_neurons, num_neurons) with cosine
        similarities. Diagonal is 1.0.
    """
    # Normalize rows to unit vectors
    norms = weight_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
    normalized = weight_matrix / norms

    # Cosine similarity = dot product of normalized vectors
    sim_matrix = normalized @ normalized.t()
    return sim_matrix


def symmetry_metric(weight_matrix: torch.Tensor) -> float:
    """Compute mean pairwise cosine similarity (excluding diagonal).

    A value of 1.0 means all neurons are identical (fully symmetric).
    A value near 0.0 means neurons are approximately orthogonal (fully broken).

    Args:
        weight_matrix: Shape (num_neurons, input_dim).

    Returns:
        Mean off-diagonal pairwise cosine similarity as a float.
    """
    n = weight_matrix.size(0)
    if n < 2:
        return 1.0

    sim = pairwise_cosine_similarity(weight_matrix)

    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    off_diag = sim[mask]

    return off_diag.mean().item()


def compute_symmetry_trajectory(
    weight_snapshots: List[torch.Tensor],
) -> List[float]:
    """Compute symmetry metric at each snapshot.

    Args:
        weight_snapshots: List of weight matrices at different training steps.

    Returns:
        List of symmetry metric values, one per snapshot.
    """
    return [symmetry_metric(w) for w in weight_snapshots]
