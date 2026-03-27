"""Calibration metrics: ECE, Brier score, reliability diagram data.

Implements standard calibration evaluation following Guo et al. (2017)
"On Calibration of Modern Neural Networks".
"""

import numpy as np
from typing import Tuple


def expected_calibration_error(probs: np.ndarray,
                               labels: np.ndarray,
                               n_bins: int = 10) -> Tuple[float, dict]:
    """Compute Expected Calibration Error (ECE).

    ECE = sum_{b=1}^{B} (n_b / N) * |acc(b) - conf(b)|

    where acc(b) is accuracy in bin b and conf(b) is mean confidence in bin b.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes).
        labels: True labels, shape (n_samples,).
        n_bins: Number of confidence bins.

    Returns:
        Tuple of (ece_value, bin_data) where bin_data contains per-bin
        accuracy, confidence, and count for reliability diagrams.
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies_mask = (predictions == labels).astype(np.float64)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    ece = 0.0
    n_total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        if i == n_bins - 1:
            # Last bin includes right boundary
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)

        n_in_bin = int(in_bin.sum())
        bin_counts.append(n_in_bin)

        if n_in_bin > 0:
            bin_acc = float(accuracies_mask[in_bin].mean())
            bin_conf = float(confidences[in_bin].mean())
            bin_accs.append(bin_acc)
            bin_confs.append(bin_conf)
            ece += (n_in_bin / n_total) * abs(bin_acc - bin_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    bin_data = {
        'bin_accs': bin_accs,
        'bin_confs': bin_confs,
        'bin_counts': bin_counts,
        'bin_edges': bin_boundaries.tolist(),
    }

    return float(ece), bin_data


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute multi-class Brier score.

    Brier = (1/N) * sum_i sum_c (p_{i,c} - y_{i,c})^2

    where y_{i,c} is 1 if sample i has label c, else 0.

    Lower is better. Range: [0, 2].

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes).
        labels: True labels, shape (n_samples,).

    Returns:
        Brier score as float.
    """
    n_samples, n_classes = probs.shape
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(n_samples), labels] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def confidence_histogram(probs: np.ndarray,
                         n_bins: int = 10) -> dict:
    """Compute confidence histogram data.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes).
        n_bins: Number of bins.

    Returns:
        Dict with 'bin_edges', 'counts', 'mean_confidence'.
    """
    confidences = np.max(probs, axis=1)
    counts, bin_edges = np.histogram(confidences, bins=n_bins, range=(0.0, 1.0))

    return {
        'bin_edges': bin_edges.tolist(),
        'counts': counts.tolist(),
        'mean_confidence': float(confidences.mean()),
    }


def accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
    """Compute classification accuracy.

    Args:
        probs: Predicted probabilities, shape (n_samples, n_classes).
        labels: True labels, shape (n_samples,).

    Returns:
        Accuracy as float in [0, 1].
    """
    predictions = np.argmax(probs, axis=1)
    return float(np.mean(predictions == labels))
