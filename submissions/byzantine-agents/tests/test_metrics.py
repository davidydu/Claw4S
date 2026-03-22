"""Tests for metric computation functions."""

import pytest

from src.metrics import (
    byzantine_threshold,
    byzantine_amplification,
    resilience_score,
)


class TestByzantineThreshold:
    def test_drops_at_known_point(self):
        fracs = [0.0, 0.1, 0.2, 0.33, 0.5]
        accs = [0.90, 0.80, 0.60, 0.40, 0.20]
        thresh = byzantine_threshold(fracs, accs)
        # Should interpolate between f=0.2 (acc=0.60) and f=0.33 (acc=0.40)
        assert 0.2 < thresh < 0.33

    def test_never_drops_returns_one(self):
        fracs = [0.0, 0.1, 0.2, 0.33, 0.5]
        accs = [0.90, 0.85, 0.80, 0.75, 0.70]
        assert byzantine_threshold(fracs, accs) == 1.0

    def test_already_below_returns_zero(self):
        fracs = [0.0, 0.1, 0.2]
        accs = [0.40, 0.30, 0.20]
        assert byzantine_threshold(fracs, accs) == 0.0

    def test_exact_cutoff(self):
        fracs = [0.0, 0.1, 0.2, 0.3]
        accs = [0.90, 0.70, 0.50, 0.30]
        thresh = byzantine_threshold(fracs, accs, cutoff=0.50)
        assert abs(thresh - 0.2) < 1e-9

    def test_symmetric_fractions(self):
        """Symmetric input should produce symmetric threshold."""
        fracs = [0.0, 0.25, 0.50]
        accs = [1.0, 0.50, 0.0]
        thresh = byzantine_threshold(fracs, accs)
        assert abs(thresh - 0.25) < 1e-9


class TestByzantineAmplification:
    def test_strategic_worse_than_random(self):
        amp = byzantine_amplification(
            accuracy_at_f_strategic=0.40,
            accuracy_at_f_random=0.60,
            baseline_accuracy=0.90,
        )
        # (0.90 - 0.40) / (0.90 - 0.60) = 0.50 / 0.30 = 1.667
        assert abs(amp - 5 / 3) < 1e-6

    def test_equal_degradation(self):
        amp = byzantine_amplification(0.50, 0.50, 0.90)
        assert abs(amp - 1.0) < 1e-6

    def test_no_degradation_from_random(self):
        """When random causes no degradation, amplification is large."""
        amp = byzantine_amplification(0.50, 0.90, 0.90)
        assert amp > 100  # near-infinite amplification


class TestResilienceScore:
    def test_perfect_resilience(self):
        fracs = [0.0, 0.25, 0.50]
        accs = [1.0, 1.0, 1.0]
        assert abs(resilience_score(fracs, accs) - 1.0) < 1e-9

    def test_zero_resilience(self):
        fracs = [0.0, 0.25, 0.50]
        accs = [0.0, 0.0, 0.0]
        assert abs(resilience_score(fracs, accs)) < 1e-9

    def test_linear_decay(self):
        fracs = [0.0, 0.50]
        accs = [1.0, 0.0]
        # Area under triangle = 0.5 * 0.5 * 1.0 = 0.25; max = 0.5 * 1.0 = 0.5
        # score = 0.25 / 0.5 = 0.5
        assert abs(resilience_score(fracs, accs) - 0.5) < 1e-9

    def test_monotonic(self):
        """Higher accuracies -> higher resilience."""
        fracs = [0.0, 0.25, 0.50]
        low = resilience_score(fracs, [0.60, 0.40, 0.20])
        high = resilience_score(fracs, [0.90, 0.70, 0.50])
        assert high > low
