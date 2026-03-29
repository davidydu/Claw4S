"""Tests for phase transition detection and lag analysis."""

import numpy as np
from src.analysis import (
    smooth,
    detect_transition_epoch,
    detect_peak_epoch,
    compute_gradient_norm_rate,
    cross_correlation_lag,
)


class TestSmooth:
    def test_preserves_length(self):
        signal = list(np.random.randn(100))
        result = smooth(signal, window=11)
        assert len(result) == len(signal)

    def test_short_signal(self):
        signal = [1.0, 2.0, 3.0]
        result = smooth(signal, window=51)
        assert len(result) == len(signal)

    def test_reduces_noise(self):
        clean = np.sin(np.linspace(0, 4 * np.pi, 200))
        noisy = clean + np.random.randn(200) * 0.3
        smoothed = smooth(noisy.tolist(), window=21)
        # Smoothed should be closer to clean than noisy is
        err_noisy = np.mean((noisy - clean) ** 2)
        err_smooth = np.mean((smoothed - clean) ** 2)
        assert err_smooth < err_noisy


class TestDetectTransition:
    def test_increasing_signal(self):
        """Step function: transition should be at the step."""
        signal = [0.0] * 50 + [1.0] * 50
        epochs = list(range(100))
        result = detect_transition_epoch(signal, epochs, direction="increase")
        # Transition should be near epoch 50
        assert 40 <= result["transition_epoch"] <= 60

    def test_decreasing_signal(self):
        signal = [1.0] * 50 + [0.0] * 50
        epochs = list(range(100))
        result = detect_transition_epoch(signal, epochs, direction="decrease")
        assert 40 <= result["transition_epoch"] <= 60


class TestComputeGradientNormRate:
    def test_combined_norm(self):
        grad_norms = {
            "layer1": [3.0, 4.0],
            "layer2": [4.0, 3.0],
        }
        combined = compute_gradient_norm_rate(grad_norms)
        assert len(combined) == 2
        assert abs(combined[0] - 5.0) < 1e-6  # sqrt(9 + 16) = 5
        assert abs(combined[1] - 5.0) < 1e-6  # sqrt(16 + 9) = 5


class TestCrossCorrelationLag:
    def test_identical_signals(self):
        signal = list(np.sin(np.linspace(0, 4 * np.pi, 200)))
        result = cross_correlation_lag(signal, signal, max_lag=50)
        # Lag should be at or very near zero for identical signals
        assert abs(result["best_lag"]) <= 1

    def test_shifted_signal(self):
        """Signal B is signal A shifted by 10 samples."""
        n = 200
        a = list(np.sin(np.linspace(0, 4 * np.pi, n)))
        shift = 10
        b = [0.0] * shift + a[:n - shift]
        result = cross_correlation_lag(a, b, max_lag=50)
        # Best lag should be close to shift
        assert abs(result["best_lag"] - shift) <= 3

    def test_flat_signals_prefer_zero_lag(self):
        """Flat signals should resolve ties to 0 lag, not an edge lag."""
        a = [1.0] * 100
        b = [1.0] * 100
        result = cross_correlation_lag(a, b, max_lag=10)
        assert result["best_lag"] == 0


class TestDetectPeakEpoch:
    def test_single_point_signal(self):
        """Single-point trajectories should not crash peak detection."""
        result = detect_peak_epoch([0.5], [7])
        assert result["transition_epoch"] == 7
        assert result["transition_idx"] == 0
