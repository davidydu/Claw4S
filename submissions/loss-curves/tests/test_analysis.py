"""Tests for analysis-level summaries."""

from src.analysis import _analyze_universality


def test_best_form_by_task_uses_total_aic_across_sizes():
    runs = [
        {
            "task": "synthetic",
            "best_form": "power_law",
            "fits": [
                {
                    "form": "power_law",
                    "aic": 10.0,
                    "converged": True,
                    "params": {"beta": 0.5},
                },
                {
                    "form": "stretched_exp",
                    "aic": 11.0,
                    "converged": True,
                    "params": {"gamma": 0.7},
                },
            ],
        },
        {
            "task": "synthetic",
            "best_form": "power_law",
            "fits": [
                {
                    "form": "power_law",
                    "aic": 12.0,
                    "converged": True,
                    "params": {"beta": 0.6},
                },
                {
                    "form": "stretched_exp",
                    "aic": 13.0,
                    "converged": True,
                    "params": {"gamma": 0.8},
                },
            ],
        },
        {
            "task": "synthetic",
            "best_form": "stretched_exp",
            "fits": [
                {
                    "form": "power_law",
                    "aic": 100.0,
                    "converged": True,
                    "params": {"beta": 0.9},
                },
                {
                    "form": "stretched_exp",
                    "aic": -200.0,
                    "converged": True,
                    "params": {"gamma": 1.1},
                },
            ],
        },
    ]

    universality = _analyze_universality(runs)

    assert universality["best_form_by_task"]["synthetic"] == "stretched_exp"
