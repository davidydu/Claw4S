"""Tests for analysis and sigmoid fitting."""

import numpy as np
import pytest

from src.experiment import ExperimentConfig, RunResult
from src.analysis import (
    AggregatedPoint,
    SigmoidFit,
    _sigmoid,
    aggregate_results,
    build_performance_payload,
    build_results_payload,
    compute_findings,
    fit_sigmoid_curve,
)


def _make_run(pf: float, hw: int, seed: int, test_acc: float) -> RunResult:
    """Helper to create a RunResult with controlled test accuracy."""
    return RunResult(
        poison_fraction=pf,
        hidden_width=hw,
        seed=seed,
        train_accuracy=test_acc + 0.05,
        test_accuracy=test_acc,
        train_clean_accuracy=test_acc - 0.02,
        generalization_gap=0.05,
        final_loss=0.5,
        elapsed_seconds=0.1,
    )


class TestAggregateResults:
    """Tests for aggregate_results."""

    def test_groups_correctly(self):
        runs = [
            _make_run(0.0, 32, 42, 0.9),
            _make_run(0.0, 32, 123, 0.88),
            _make_run(0.0, 32, 7, 0.92),
            _make_run(0.1, 32, 42, 0.7),
        ]
        agg = aggregate_results(runs)
        assert len(agg) == 2  # Two unique (pf, hw) pairs

    def test_mean_std(self):
        runs = [
            _make_run(0.0, 64, 42, 0.8),
            _make_run(0.0, 64, 123, 0.9),
            _make_run(0.0, 64, 7, 0.85),
        ]
        agg = aggregate_results(runs)
        assert len(agg) == 1
        assert abs(agg[0].test_acc_mean - 0.85) < 1e-6
        assert agg[0].test_acc_std > 0
        assert agg[0].n_seeds == 3


class TestSigmoidFit:
    """Tests for sigmoid curve fitting."""

    def test_fit_perfect_sigmoid(self):
        """Fit a known sigmoid curve."""
        x = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
        y = _sigmoid(x, L=0.7, k=15.0, x0=0.2, b=0.2)

        runs = []
        for i, (pf, acc) in enumerate(zip(x, y)):
            for seed in [42, 123, 7]:
                runs.append(_make_run(pf, 64, seed, acc + np.random.RandomState(seed).randn() * 0.001))

        agg = aggregate_results(runs)
        fit = fit_sigmoid_curve(agg, hidden_width=64)

        assert fit.r_squared > 0.95
        assert abs(fit.x0 - 0.2) < 0.05
        assert fit.hidden_width == 64

    def test_fit_returns_threshold(self):
        """Check that threshold is computed."""
        x = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5])
        y = _sigmoid(x, L=0.7, k=12.0, x0=0.2, b=0.2)

        runs = []
        for pf, acc in zip(x, y):
            for seed in [42, 123, 7]:
                runs.append(_make_run(pf, 128, seed, acc))

        agg = aggregate_results(runs)
        fit = fit_sigmoid_curve(agg, hidden_width=128)
        assert 0 < fit.threshold_midpoint < 0.6

    def test_fit_threshold_supports_higher_poison_ranges(self):
        """Threshold search should work beyond 60% poison."""
        x = np.array([0.0, 0.1, 0.2, 0.4, 0.6, 0.75, 0.85, 0.9])
        y = _sigmoid(x, L=0.7, k=11.0, x0=0.72, b=0.2)

        runs = []
        for pf, acc in zip(x, y):
            for seed in [42, 123, 7]:
                runs.append(_make_run(pf, 64, seed, acc))

        agg = aggregate_results(runs)
        fit = fit_sigmoid_curve(agg, hidden_width=64)
        assert 0.65 < fit.threshold_midpoint < 0.85


class TestSigmoidFunction:
    """Tests for the sigmoid helper."""

    def test_midpoint(self):
        x = np.array([0.2])
        y = _sigmoid(x, L=1.0, k=10.0, x0=0.2, b=0.0)
        assert abs(y[0] - 0.5) < 1e-6

    def test_limits(self):
        x_low = np.array([-10.0])
        x_high = np.array([10.0])
        y_low = _sigmoid(x_low, L=1.0, k=10.0, x0=0.0, b=0.0)
        y_high = _sigmoid(x_high, L=1.0, k=10.0, x0=0.0, b=0.0)
        assert y_low[0] > 0.99  # Far left -> L + b
        assert y_high[0] < 0.01 + 0.0  # Far right -> b


class TestResultSerialization:
    """Tests for exporting deterministic scientific results."""

    def test_results_payload_excludes_timing_fields(self):
        config = ExperimentConfig(
            poison_fractions=(0.0,),
            hidden_widths=(32,),
            seeds=(42,),
        )
        runs = [_make_run(0.0, 32, 42, 0.9)]
        agg = [
            AggregatedPoint(
                poison_fraction=0.0,
                hidden_width=32,
                test_acc_mean=0.9,
                test_acc_std=0.0,
                train_acc_mean=0.95,
                train_acc_std=0.0,
                train_clean_acc_mean=0.88,
                train_clean_acc_std=0.0,
                gen_gap_mean=0.05,
                gen_gap_std=0.0,
                n_seeds=1,
            )
        ]
        fits = [
            SigmoidFit(
                hidden_width=32,
                L=0.7,
                k=4.8,
                x0=0.18,
                b=0.2,
                r_squared=0.99,
                threshold_midpoint=0.43,
            )
        ]
        findings = compute_findings(agg, fits)

        payload = build_results_payload(config, runs, agg, fits, findings)

        assert "elapsed_seconds" not in payload["runs"][0]
        assert "total_time_seconds" not in payload["metadata"]

    def test_performance_payload_keeps_runtime_metadata(self):
        runs = [
            _make_run(0.0, 32, 42, 0.9),
            _make_run(0.5, 32, 42, 0.5),
        ]

        payload = build_performance_payload(runs, total_time_seconds=12.5)

        assert payload["total_time_seconds"] == pytest.approx(12.5)
        assert payload["n_runs"] == 2
        assert payload["mean_run_time_seconds"] == pytest.approx(0.1)


class TestFindingsGeneralizability:
    """Tests that findings logic adapts to non-default width sets."""

    def test_findings_cover_dynamic_widths(self):
        agg = [
            AggregatedPoint(0.0, 16, 0.80, 0.01, 0.83, 0.01, 0.79, 0.01, 0.03, 0.01, 3),
            AggregatedPoint(0.5, 16, 0.45, 0.02, 0.55, 0.02, 0.50, 0.02, 0.10, 0.02, 3),
            AggregatedPoint(0.0, 48, 0.82, 0.01, 0.85, 0.01, 0.81, 0.01, 0.03, 0.01, 3),
            AggregatedPoint(0.5, 48, 0.40, 0.02, 0.58, 0.02, 0.49, 0.02, 0.18, 0.02, 3),
        ]
        fits = [
            SigmoidFit(16, 0.6, 5.0, 0.30, 0.2, 0.95, 0.41),
            SigmoidFit(48, 0.6, 6.0, 0.25, 0.2, 0.96, 0.36),
        ]

        findings = compute_findings(agg, fits)

        assert set(findings["clean_test_accuracy"].keys()) == {16, 48}
        assert set(findings["gen_gap_at_50pct_poison"].keys()) == {16, 48}
