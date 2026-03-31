"""Tests for spectral signature detection."""

import numpy as np
import pytest

from src.spectral import (
    compute_spectral_scores,
    compute_detection_auc,
    compute_eigenvalue_ratio,
)


class TestComputeSpectralScores:
    """Tests for spectral score computation."""

    def test_output_shapes(self):
        acts = np.random.randn(100, 32).astype(np.float32)
        scores, eigenvalues, top_eigvec = compute_spectral_scores(acts)
        assert scores.shape == (100,)
        assert eigenvalues.shape == (32,)
        assert top_eigvec.shape == (32,)

    def test_scores_nonnegative(self):
        acts = np.random.randn(100, 32).astype(np.float32)
        scores, _, _ = compute_spectral_scores(acts)
        assert (scores >= 0).all()

    def test_eigenvalues_sorted_descending(self):
        acts = np.random.randn(100, 32).astype(np.float32)
        _, eigenvalues, _ = compute_spectral_scores(acts)
        for i in range(len(eigenvalues) - 1):
            assert eigenvalues[i] >= eigenvalues[i + 1]

    def test_dominant_direction_detected(self):
        """If we add a strong signal along one direction, top eigenvector should align."""
        rng = np.random.RandomState(42)
        acts = rng.randn(100, 10).astype(np.float32) * 0.1
        # Add dominant signal along first axis for some samples
        acts[:20, 0] += 10.0
        _, eigenvalues, top_eigvec = compute_spectral_scores(acts)
        # Top eigenvector should have large component along axis 0
        assert abs(top_eigvec[0]) > 0.5


class TestComputeDetectionAUC:
    """Tests for AUC computation."""

    def test_perfect_detection(self):
        scores = np.array([10.0, 9.0, 8.0, 0.1, 0.2, 0.3])
        mask = np.array([True, True, True, False, False, False])
        auc = compute_detection_auc(scores, mask)
        assert auc == 1.0

    def test_random_detection(self):
        rng = np.random.RandomState(42)
        scores = rng.randn(1000)
        mask = rng.rand(1000) > 0.5
        auc = compute_detection_auc(scores, mask)
        # Random scores should give AUC near 0.5
        assert 0.35 < auc < 0.65

    def test_no_positives_raises(self):
        scores = np.array([1.0, 2.0, 3.0])
        mask = np.array([False, False, False])
        with pytest.raises(ValueError, match="No poisoned samples"):
            compute_detection_auc(scores, mask)

    def test_no_negatives_raises(self):
        scores = np.array([1.0, 2.0, 3.0])
        mask = np.array([True, True, True])
        with pytest.raises(ValueError, match="No clean samples"):
            compute_detection_auc(scores, mask)


class TestComputeEigenvalueRatio:
    """Tests for eigenvalue ratio computation."""

    def test_normal_case(self):
        eigenvalues = np.array([10.0, 2.0, 1.0])
        ratio = compute_eigenvalue_ratio(eigenvalues)
        assert ratio == pytest.approx(5.0)

    def test_single_eigenvalue(self):
        eigenvalues = np.array([5.0])
        ratio = compute_eigenvalue_ratio(eigenvalues)
        assert ratio == float("inf")

    def test_zero_second(self):
        eigenvalues = np.array([5.0, 0.0])
        ratio = compute_eigenvalue_ratio(eigenvalues)
        assert ratio == float("inf")
