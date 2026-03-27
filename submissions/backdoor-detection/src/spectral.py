"""Spectral signature detection of backdoor attacks (Tran et al. 2018).

Implements the core spectral analysis: compute the covariance matrix of
penultimate-layer activations, extract the top eigenvector, and use it
to score each sample. Poisoned samples cluster along the top eigenvector
direction, producing higher outlier scores.
"""

import numpy as np
from scipy import linalg


def compute_spectral_scores(
    activations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute outlier scores via spectral signature analysis.

    Centers the activations, computes the covariance matrix, extracts
    the top eigenvector, and projects each sample onto it. The squared
    projection magnitudes serve as outlier scores.

    Args:
        activations: Activation matrix of shape (n_samples, hidden_dim).

    Returns:
        scores: Outlier scores of shape (n_samples,).
        eigenvalues: All eigenvalues of the covariance matrix, sorted descending.
        top_eigvec: Top eigenvector of the covariance matrix.
    """
    # Center activations
    mean_act = activations.mean(axis=0)
    centered = activations - mean_act

    # Covariance matrix
    cov = np.dot(centered.T, centered) / len(centered)

    # Eigendecomposition (symmetric matrix)
    eigenvalues, eigenvectors = linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top eigenvector
    top_eigvec = eigenvectors[:, 0]

    # Outlier score: squared projection onto top eigenvector
    projections = np.dot(centered, top_eigvec)
    scores = projections ** 2

    return scores, eigenvalues, top_eigvec


def compute_detection_auc(
    scores: np.ndarray,
    poison_mask: np.ndarray,
) -> float:
    """Compute AUC for detecting poisoned samples using spectral scores.

    Args:
        scores: Outlier scores of shape (n_samples,).
        poison_mask: Boolean array where True = poisoned sample.

    Returns:
        AUC score (0 to 1). Values near 1 mean the spectral method
        can reliably distinguish poisoned from clean samples.

    Raises:
        ValueError: If poison_mask has no positive or no negative samples.
    """
    n_positive = poison_mask.sum()
    n_negative = (~poison_mask).sum()
    if n_positive == 0:
        raise ValueError("No poisoned samples in mask; cannot compute AUC.")
    if n_negative == 0:
        raise ValueError("No clean samples in mask; cannot compute AUC.")

    # Manual AUC via the Mann-Whitney U statistic to avoid sklearn dependency.
    # AUC = P(score_positive > score_negative).
    pos_scores = scores[poison_mask]
    neg_scores = scores[~poison_mask]
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)

    # Count how many (positive, negative) pairs have pos > neg
    # Use broadcasting for efficiency
    comparisons = (pos_scores[:, None] > neg_scores[None, :]).sum()
    ties = (pos_scores[:, None] == neg_scores[None, :]).sum()
    auc = (comparisons + 0.5 * ties) / (n_pos * n_neg)
    return float(auc)


def compute_eigenvalue_ratio(eigenvalues: np.ndarray) -> float:
    """Compute the ratio of the top eigenvalue to the second eigenvalue.

    A high ratio indicates that one direction dominates the activation
    variance, consistent with a backdoor signal.

    Args:
        eigenvalues: Eigenvalues sorted in descending order.

    Returns:
        Ratio of first to second eigenvalue.
    """
    if len(eigenvalues) < 2:
        return float("inf")
    if eigenvalues[1] <= 0:
        return float("inf")
    return float(eigenvalues[0] / eigenvalues[1])
