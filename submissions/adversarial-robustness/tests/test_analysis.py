"""Tests for analysis and visualization."""

import math
import os
import tempfile

import numpy as np

from src.analysis import (
    compute_robustness_gaps,
    compute_summary_statistics,
    summarize_results_by_dataset,
)


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


def _make_multiseed_dataset_results():
    """Create results where combined seeds change the reported trend."""
    results = []
    widths = [16, 32, 64]
    epsilons = [0.1, 0.2]

    circle_specs = {
        0: {
            16: {"clean": 0.90, "fgsm_base_gap": 0.30},
            32: {"clean": 0.91, "fgsm_base_gap": 0.25},
            64: {"clean": 0.92, "fgsm_base_gap": 0.20},
        },
        1: {
            16: {"clean": 0.80, "fgsm_base_gap": 0.10},
            32: {"clean": 0.82, "fgsm_base_gap": 0.18},
            64: {"clean": 0.84, "fgsm_base_gap": 0.26},
        },
    }

    moons_specs = {
        0: {
            16: {"clean": 0.96, "fgsm_base_gap": 0.22},
            32: {"clean": 0.97, "fgsm_base_gap": 0.20},
            64: {"clean": 0.98, "fgsm_base_gap": 0.18},
        },
        1: {
            16: {"clean": 0.94, "fgsm_base_gap": 0.24},
            32: {"clean": 0.95, "fgsm_base_gap": 0.21},
            64: {"clean": 0.96, "fgsm_base_gap": 0.19},
        },
    }

    for dataset, specs in [("circles", circle_specs), ("moons", moons_specs)]:
        for seed, seed_specs in specs.items():
            for width in widths:
                clean = seed_specs[width]["clean"]
                fgsm_base_gap = seed_specs[width]["fgsm_base_gap"]
                for eps in epsilons:
                    fgsm_gap = fgsm_base_gap + (0.0 if eps == 0.1 else 0.05)
                    pgd_gap = fgsm_gap + 0.02
                    results.append({
                        "dataset": dataset,
                        "seed": seed,
                        "hidden_width": width,
                        "param_count": width * width + 6 * width + 2,
                        "clean_acc": clean,
                        "epsilon": eps,
                        "fgsm_acc": clean - fgsm_gap,
                        "pgd_acc": clean - pgd_gap,
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

    def test_trend_stats_include_uncertainty_and_significance(self):
        results = _make_sample_results()
        augmented = compute_robustness_gaps(results)
        summary = compute_summary_statistics(augmented)

        fgsm_trend = summary["trend_fgsm_gap"]
        assert math.isclose(fgsm_trend["pearson_r"], summary["corr_logparams_fgsm_gap"])
        assert 0.0 <= fgsm_trend["pearson_p_value"] <= 1.0
        assert 0.0 <= fgsm_trend["spearman_p_value"] <= 1.0
        assert -1.0 <= fgsm_trend["spearman_rho"] <= 1.0
        assert len(fgsm_trend["pearson_r_ci95"]) == 2
        ci_low, ci_high = fgsm_trend["pearson_r_ci95"]
        assert ci_low <= fgsm_trend["pearson_r"] <= ci_high

        pgd_trend = summary["trend_pgd_gap"]
        assert math.isclose(pgd_trend["pearson_r"], summary["corr_logparams_pgd_gap"])
        assert len(pgd_trend["pearson_r_ci95"]) == 2

    def test_clean_acc_averaged_across_replicates(self):
        results = _make_multiseed_dataset_results()
        circles = [r for r in results if r["dataset"] == "circles"]
        augmented = compute_robustness_gaps(circles)
        summary = compute_summary_statistics(augmented)

        # Width 16 appears with clean_acc 0.90 and 0.80 across two seeds.
        assert math.isclose(summary["per_width"][16]["clean_acc"], 0.85)


class TestSummarizeResultsByDataset:
    def test_uses_all_seeds_for_dataset_correlations(self):
        results = _make_multiseed_dataset_results()
        augmented = compute_robustness_gaps(results)

        summaries = summarize_results_by_dataset(augmented)
        circles = summaries["circles"]

        widths = sorted(circles["widths"])
        log_params = np.log10([circles["per_width"][w]["param_count"] for w in widths])
        mean_fgsm_gaps = np.array([circles["per_width"][w]["mean_fgsm_gap"]
                                   for w in widths])
        expected_corr = float(np.corrcoef(log_params, mean_fgsm_gaps)[0, 1])

        first_seed_results = [r for r in augmented
                              if r["dataset"] == "circles" and r["seed"] == 0]
        first_seed_summary = compute_summary_statistics(first_seed_results)

        assert math.isclose(circles["corr_logparams_fgsm_gap"], expected_corr)
        assert circles["n_experiments"] == 12
        assert not math.isclose(
            circles["corr_logparams_fgsm_gap"],
            first_seed_summary["corr_logparams_fgsm_gap"],
        )
