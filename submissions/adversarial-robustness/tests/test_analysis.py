"""Tests for analysis and visualization."""

import os
import tempfile

from src.analysis import compute_robustness_gaps, compute_summary_statistics


def _make_sample_results():
    """Create sample results for testing.

    Simulates data where larger models have higher clean accuracy but also
    larger robustness gaps (more vulnerable to adversarial perturbations).
    """
    results = []
    # Vary clean_acc and gaps with width to produce non-degenerate correlations
    width_clean = {16: 0.88, 32: 0.92, 64: 0.95}
    for w in [16, 32, 64]:
        clean = width_clean[w]
        for eps in [0.1, 0.2]:
            # Larger widths lose more accuracy under attack
            fgsm_drop = eps * (1.0 + w / 64.0)
            pgd_drop = eps * (1.5 + w / 64.0)
            results.append({
                "hidden_width": w,
                "param_count": w * w + 6 * w + 2,
                "clean_acc": clean,
                "epsilon": eps,
                "fgsm_acc": max(0.0, clean - fgsm_drop),
                "pgd_acc": max(0.0, clean - pgd_drop),
            })
    return results


class TestComputeRobustnessGaps:
    def test_adds_gap_fields(self):
        results = _make_sample_results()
        augmented = compute_robustness_gaps(results)
        for r in augmented:
            assert "fgsm_gap" in r
            assert "pgd_gap" in r

    def test_gap_values_correct(self):
        results = _make_sample_results()
        augmented = compute_robustness_gaps(results)
        for r in augmented:
            assert abs(r["fgsm_gap"] - (r["clean_acc"] - r["fgsm_acc"])) < 1e-6
            assert abs(r["pgd_gap"] - (r["clean_acc"] - r["pgd_acc"])) < 1e-6

    def test_gaps_non_negative(self):
        results = _make_sample_results()
        augmented = compute_robustness_gaps(results)
        for r in augmented:
            assert r["fgsm_gap"] >= -1e-6
            assert r["pgd_gap"] >= -1e-6


class TestComputeSummaryStatistics:
    def test_summary_keys(self):
        results = _make_sample_results()
        augmented = compute_robustness_gaps(results)
        summary = compute_summary_statistics(augmented)
        assert "widths" in summary
        assert "epsilons" in summary
        assert "per_width" in summary
        assert "n_experiments" in summary

    def test_per_width_keys(self):
        results = _make_sample_results()
        augmented = compute_robustness_gaps(results)
        summary = compute_summary_statistics(augmented)
        for w in summary["widths"]:
            pw = summary["per_width"][w]
            assert "clean_acc" in pw
            assert "mean_fgsm_gap" in pw
            assert "mean_pgd_gap" in pw
            assert "std_fgsm_gap" in pw

    def test_correlation_computed(self):
        results = _make_sample_results()
        augmented = compute_robustness_gaps(results)
        summary = compute_summary_statistics(augmented)
        assert summary["corr_logparams_fgsm_gap"] is not None
        assert -1.0 <= summary["corr_logparams_fgsm_gap"] <= 1.0
