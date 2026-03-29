"""Tests for src/report.py."""

from src.report import generate_report


def test_report_includes_uncertainty_and_task_stratified_sections():
    """Report renders CIs and per-task correlation summaries."""
    results = {
        "config": {
            "hidden_widths": [32, 64],
            "n_epochs": 100,
            "mod_add_lr": 0.01,
            "mod_add_wd": 1.0,
            "reg_lr": 0.01,
            "reg_wd": 0.1,
            "seed": 42,
        },
        "experiment_summaries": [
            {
                "task": "modular_addition_mod97",
                "hidden_dim": 32,
                "final_dead_frac": 0.0,
                "final_zero_frac": 0.45,
                "zero_frac_change": -0.05,
                "final_test_acc": 0.55,
                "gen_gap": 0.45,
            },
            {
                "task": "nonlinear_regression",
                "hidden_dim": 32,
                "final_dead_frac": 0.0,
                "final_zero_frac": 0.60,
                "zero_frac_change": 0.04,
                "final_test_acc": 0.90,
                "gen_gap": 0.05,
            },
        ],
        "correlations": {
            "zero_frac_vs_gen_gap": {
                "rho": -0.8,
                "p_value": 0.02,
                "n": 8,
                "ci_low": -0.95,
                "ci_high": -0.30,
            },
        },
        "correlations_by_task": {
            "modular_addition_mod97": {
                "zero_frac_vs_gen_gap": {
                    "rho": -0.6,
                    "p_value": 0.25,
                    "n": 4,
                    "ci_low": -1.0,
                    "ci_high": 0.4,
                },
            },
            "nonlinear_regression": {
                "zero_frac_vs_gen_gap": {
                    "rho": -0.4,
                    "p_value": 0.60,
                    "n": 4,
                    "ci_low": -1.0,
                    "ci_high": 0.8,
                },
            },
        },
        "grokking_analysis": [
            {
                "hidden_dim": 32,
                "grokking_detected": False,
                "max_test_acc": 0.55,
                "zero_frac_initial": 0.50,
                "zero_frac_final": 0.45,
            },
        ],
    }

    report = generate_report(results)

    assert "95% CI" in report
    assert "Task-Stratified Correlations" in report
    assert "modular addition mod97" in report.lower()
