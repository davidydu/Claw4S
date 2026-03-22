"""Tests for analysis utilities."""

import pytest

from src.analysis import (
    find_interpolation_peak,
    find_minimum_test_loss,
    compute_double_descent_ratio,
    detect_double_descent,
    detect_epoch_wise_double_descent,
    compute_variance_bands,
)


def make_sweep_results(widths, test_losses):
    """Helper to create sweep-like results."""
    return [
        {"width": w, "test_loss": t, "train_loss": 0.0, "n_params": w,
         "param_ratio": w / 200}
        for w, t in zip(widths, test_losses)
    ]


class TestFindInterpolationPeak:
    def test_finds_max(self):
        results = make_sweep_results([50, 100, 200, 500], [1.0, 5.0, 100.0, 2.0])
        width, loss = find_interpolation_peak(results)
        assert width == 200
        assert loss == 100.0

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            find_interpolation_peak([])


class TestFindMinimumTestLoss:
    def test_finds_min(self):
        results = make_sweep_results([50, 100, 200, 500], [3.0, 5.0, 100.0, 2.0])
        width, loss = find_minimum_test_loss(results)
        assert width == 500
        assert loss == 2.0


class TestDoubleDescentRatio:
    def test_ratio(self):
        results = make_sweep_results([50, 100, 200, 500], [3.0, 5.0, 100.0, 2.0])
        ratio = compute_double_descent_ratio(results)
        assert ratio == pytest.approx(50.0)  # 100 / 2


class TestDetectDoubleDescent:
    def test_detects_clear_pattern(self):
        results = make_sweep_results(
            [50, 100, 150, 200, 300, 500, 1000],
            [5.0, 3.0, 10.0, 100.0, 5.0, 2.0, 1.5],
        )
        detection = detect_double_descent(results)
        assert detection["detected"] is True
        assert detection["peak_width"] == 200
        assert detection["ratio"] > 10

    def test_no_double_descent_monotone(self):
        results = make_sweep_results(
            [50, 100, 200, 500, 1000],
            [5.0, 4.0, 3.0, 2.0, 1.0],
        )
        detection = detect_double_descent(results)
        assert detection["detected"] is False

    def test_peak_at_boundary_not_detected(self):
        results = make_sweep_results(
            [50, 100, 200, 500],
            [100.0, 5.0, 3.0, 1.0],
        )
        detection = detect_double_descent(results)
        assert detection["detected"] is False

    def test_too_few_points(self):
        results = make_sweep_results([50, 100], [1.0, 2.0])
        detection = detect_double_descent(results)
        assert detection["detected"] is False


class TestDetectEpochWiseDoubleDescent:
    def test_detects_pattern(self):
        epochs = [100, 200, 300, 400, 500, 600, 700, 800]
        test_losses = [5.0, 3.0, 2.0, 4.0, 6.0, 5.0, 3.0, 2.0]
        detection = detect_epoch_wise_double_descent(epochs, test_losses)
        assert detection["detected"] is True

    def test_no_pattern_monotone(self):
        epochs = [100, 200, 300, 400, 500]
        test_losses = [5.0, 4.0, 3.0, 2.0, 1.0]
        detection = detect_epoch_wise_double_descent(epochs, test_losses)
        assert detection["detected"] is False

    def test_too_few_points(self):
        epochs = [100, 200]
        test_losses = [5.0, 3.0]
        detection = detect_epoch_wise_double_descent(epochs, test_losses)
        assert detection["detected"] is False


class TestComputeVarianceBands:
    def test_basic(self):
        variance_results = [
            {"seed": 42, "results": make_sweep_results([50, 100, 200], [3.0, 5.0, 100.0])},
            {"seed": 99, "results": make_sweep_results([50, 100, 200], [4.0, 6.0, 80.0])},
        ]
        stats = compute_variance_bands(variance_results)
        assert stats["widths"] == [50, 100, 200]
        assert stats["n_seeds"] == 2
        assert len(stats["mean_test_loss"]) == 3
        assert len(stats["std_test_loss"]) == 3

    def test_empty(self):
        stats = compute_variance_bands([])
        assert stats["widths"] == []
        assert stats["n_seeds"] == 0
