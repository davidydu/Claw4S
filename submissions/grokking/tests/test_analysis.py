"""Tests for phase classification and analysis."""

from src.analysis import (
    Phase,
    aggregate_results,
    classify_phase,
    compute_grokking_gap,
)
from src.train import TrainResult


def _make_result(
    train_acc: float = 0.0,
    test_acc: float = 0.0,
    epoch_train_95: int | None = None,
    epoch_test_95: int | None = None,
) -> TrainResult:
    """Helper to create a TrainResult with specific values."""
    return TrainResult(
        final_train_acc=train_acc,
        final_test_acc=test_acc,
        epoch_train_95=epoch_train_95,
        epoch_test_95=epoch_test_95,
        total_epochs=5000,
    )


class TestClassifyPhase:
    """Tests for phase classification."""

    def test_confusion(self):
        """Low train acc -> CONFUSION."""
        r = _make_result(train_acc=0.5, test_acc=0.3)
        assert classify_phase(r) == Phase.CONFUSION

    def test_memorization(self):
        """High train, low test -> MEMORIZATION."""
        r = _make_result(
            train_acc=0.99, test_acc=0.4,
            epoch_train_95=200,
        )
        assert classify_phase(r) == Phase.MEMORIZATION

    def test_grokking(self):
        """High train, high test, large gap -> GROKKING."""
        r = _make_result(
            train_acc=0.99, test_acc=0.98,
            epoch_train_95=200, epoch_test_95=2000,
        )
        assert classify_phase(r) == Phase.GROKKING

    def test_comprehension(self):
        """High train, high test, small gap -> COMPREHENSION."""
        r = _make_result(
            train_acc=0.99, test_acc=0.98,
            epoch_train_95=200, epoch_test_95=400,
        )
        assert classify_phase(r) == Phase.COMPREHENSION

    def test_grokking_boundary(self):
        """Gap of exactly 500 -> COMPREHENSION (not grokking)."""
        r = _make_result(
            train_acc=0.99, test_acc=0.98,
            epoch_train_95=200, epoch_test_95=700,
        )
        assert classify_phase(r) == Phase.COMPREHENSION

    def test_grokking_just_above_boundary(self):
        """Gap of 501 -> GROKKING."""
        r = _make_result(
            train_acc=0.99, test_acc=0.98,
            epoch_train_95=200, epoch_test_95=701,
        )
        assert classify_phase(r) == Phase.GROKKING

    def test_train_below_threshold(self):
        """Train acc at 0.94 -> CONFUSION."""
        r = _make_result(train_acc=0.94, test_acc=0.94)
        assert classify_phase(r) == Phase.CONFUSION

    def test_train_at_threshold(self):
        """Train acc at exactly 0.95 -> not confusion (if epoch recorded)."""
        r = _make_result(
            train_acc=0.95, test_acc=0.95,
            epoch_train_95=100, epoch_test_95=200,
        )
        assert classify_phase(r) == Phase.COMPREHENSION


class TestComputeGrokkingGap:
    """Tests for grokking gap computation."""

    def test_both_reached(self):
        """Should return difference when both thresholds reached."""
        r = _make_result(epoch_train_95=200, epoch_test_95=1500)
        assert compute_grokking_gap(r) == 1300

    def test_train_not_reached(self):
        """Should return None if train threshold not reached."""
        r = _make_result(epoch_test_95=1500)
        assert compute_grokking_gap(r) is None

    def test_test_not_reached(self):
        """Should return None if test threshold not reached."""
        r = _make_result(epoch_train_95=200)
        assert compute_grokking_gap(r) is None

    def test_neither_reached(self):
        """Should return None if neither threshold reached."""
        r = _make_result()
        assert compute_grokking_gap(r) is None

    def test_zero_gap(self):
        """Gap can be zero if both hit at same epoch."""
        r = _make_result(epoch_train_95=100, epoch_test_95=100)
        assert compute_grokking_gap(r) == 0

    def test_negative_gap(self):
        """Gap can be negative if test reaches before train."""
        r = _make_result(epoch_train_95=300, epoch_test_95=100)
        assert compute_grokking_gap(r) == -200


class TestAggregateResults:
    """Tests for aggregate_results."""

    def test_counts_phases(self):
        """Should count each phase correctly."""
        results = [
            {"phase": Phase.GROKKING, "grokking_gap": 1000},
            {"phase": Phase.GROKKING, "grokking_gap": 2000},
            {"phase": Phase.MEMORIZATION, "grokking_gap": None},
            {"phase": Phase.CONFUSION, "grokking_gap": None},
        ]
        stats = aggregate_results(results)
        assert stats["phase_counts"]["grokking"] == 2
        assert stats["phase_counts"]["memorization"] == 1
        assert stats["phase_counts"]["confusion"] == 1
        assert stats["phase_counts"]["comprehension"] == 0
        assert stats["total_runs"] == 4

    def test_grokking_fraction(self):
        """Should compute correct grokking fraction."""
        results = [
            {"phase": Phase.GROKKING, "grokking_gap": 1000},
            {"phase": Phase.MEMORIZATION, "grokking_gap": None},
        ]
        stats = aggregate_results(results)
        assert stats["grokking_fraction"] == 0.5

    def test_mean_grokking_gap(self):
        """Should compute mean gap for grokking runs only."""
        results = [
            {"phase": Phase.GROKKING, "grokking_gap": 1000},
            {"phase": Phase.GROKKING, "grokking_gap": 3000},
            {"phase": Phase.MEMORIZATION, "grokking_gap": None},
        ]
        stats = aggregate_results(results)
        assert stats["mean_grokking_gap"] == 2000
        assert stats["max_grokking_gap"] == 3000

    def test_no_grokking_runs(self):
        """Should handle no grokking runs gracefully."""
        results = [
            {"phase": Phase.MEMORIZATION, "grokking_gap": None},
        ]
        stats = aggregate_results(results)
        assert stats["mean_grokking_gap"] is None
        assert stats["max_grokking_gap"] is None

    def test_empty_results(self):
        """Should handle empty results."""
        stats = aggregate_results([])
        assert stats["total_runs"] == 0
        assert stats["grokking_fraction"] == 0
