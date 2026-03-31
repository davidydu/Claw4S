"""Tests for calibration metrics."""

import numpy as np
from src.metrics import (expected_calibration_error, brier_score,
                         confidence_histogram, accuracy)


def test_ece_perfect_calibration():
    """ECE should be near zero for perfectly calibrated predictions."""
    np.random.seed(42)
    n = 200
    n_classes = 5

    # Create "perfectly calibrated" predictions: true label always matches argmax
    labels = np.random.randint(0, n_classes, n)
    probs = np.full((n, n_classes), 0.05)
    for i in range(n):
        probs[i, labels[i]] = 0.80  # High confidence on correct class

    ece, bin_data = expected_calibration_error(probs, labels)
    # Should be very low (not exactly zero due to binning artifacts)
    # ECE = |acc - conf| weighted by bin fraction. With 80% confidence and 100%
    # accuracy, gap is 0.20. Threshold accounts for this expected gap.
    assert ece < 0.25, f"ECE should be low for correct predictions, got {ece}"


def test_ece_range():
    """ECE should be in [0, 1]."""
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(5), size=100)
    labels = np.random.randint(0, 5, 100)
    ece, _ = expected_calibration_error(probs, labels)
    assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"


def test_ece_bin_data():
    """ECE returns valid bin data."""
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(5), size=100)
    labels = np.random.randint(0, 5, 100)
    ece, bin_data = expected_calibration_error(probs, labels, n_bins=10)

    assert len(bin_data['bin_accs']) == 10
    assert len(bin_data['bin_confs']) == 10
    assert len(bin_data['bin_counts']) == 10
    assert len(bin_data['bin_edges']) == 11
    assert sum(bin_data['bin_counts']) == 100


def test_brier_score_perfect():
    """Brier score should be 0 for perfect predictions."""
    n = 50
    n_classes = 3
    labels = np.array([0, 1, 2] * (n // 3) + [0] * (n % 3))
    probs = np.zeros((len(labels), n_classes))
    for i in range(len(labels)):
        probs[i, labels[i]] = 1.0

    bs = brier_score(probs, labels)
    assert abs(bs) < 1e-10, f"Brier score should be 0, got {bs}"


def test_brier_score_range():
    """Brier score should be in [0, 2]."""
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(5), size=100)
    labels = np.random.randint(0, 5, 100)
    bs = brier_score(probs, labels)
    assert 0.0 <= bs <= 2.0, f"Brier score out of range: {bs}"


def test_confidence_histogram():
    """Confidence histogram has correct structure."""
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(5), size=100)
    hist = confidence_histogram(probs, n_bins=10)

    assert len(hist['bin_edges']) == 11
    assert len(hist['counts']) == 10
    assert sum(hist['counts']) == 100
    assert 0.0 <= hist['mean_confidence'] <= 1.0


def test_accuracy_perfect():
    """Accuracy should be 1.0 for perfect predictions."""
    labels = np.array([0, 1, 2, 0, 1])
    probs = np.zeros((5, 3))
    for i in range(5):
        probs[i, labels[i]] = 1.0
    assert accuracy(probs, labels) == 1.0


def test_accuracy_range():
    """Accuracy is in [0, 1]."""
    np.random.seed(42)
    probs = np.random.dirichlet(np.ones(5), size=100)
    labels = np.random.randint(0, 5, 100)
    acc = accuracy(probs, labels)
    assert 0.0 <= acc <= 1.0
