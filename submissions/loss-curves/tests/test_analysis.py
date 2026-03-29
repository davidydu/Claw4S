"""Tests for analysis-level summaries."""

from src.analysis import (
    _analyze_universality,
    _build_provenance,
    _load_checkpoint,
    _save_checkpoint,
)


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


def test_universality_reports_delta_aic_support_strength():
    runs = [
        {
            "task": "task_a",
            "hidden_size": 32,
            "best_form": "power_law",
            "fits": [
                {
                    "form": "power_law",
                    "aic": -120.0,
                    "converged": True,
                    "params": {"beta": 0.4},
                },
                {
                    "form": "stretched_exp",
                    "aic": -108.0,
                    "converged": True,
                    "params": {"gamma": 0.8},
                },
            ],
        },
        {
            "task": "task_a",
            "hidden_size": 64,
            "best_form": "power_law",
            "fits": [
                {
                    "form": "power_law",
                    "aic": -90.0,
                    "converged": True,
                    "params": {"beta": 0.5},
                },
                {
                    "form": "stretched_exp",
                    "aic": -84.5,
                    "converged": True,
                    "params": {"gamma": 0.9},
                },
            ],
        },
        {
            "task": "task_b",
            "hidden_size": 32,
            "best_form": "stretched_exp",
            "fits": [
                {
                    "form": "stretched_exp",
                    "aic": -70.0,
                    "converged": True,
                    "params": {"gamma": 1.2},
                }
            ],
        },
    ]

    universality = _analyze_universality(runs)

    assert universality["support_counts"]["strong"] == 1
    assert universality["support_counts"]["moderate"] == 1
    assert universality["support_counts"]["weak"] == 0
    assert universality["support_counts"]["undetermined"] == 1
    assert universality["support_by_task"]["task_a"]["strong"] == 1
    assert universality["support_by_task"]["task_a"]["moderate"] == 1
    assert universality["support_by_task"]["task_b"]["undetermined"] == 1


def test_checkpoint_round_trip_and_config_guard(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    config = {
        "tasks": ["regression"],
        "hidden_sizes": [32],
        "n_epochs": 10,
        "skip_epochs": 2,
        "seed": 123,
    }
    runs = [{"task": "regression", "hidden_size": 32, "fits": []}]

    _save_checkpoint(str(checkpoint), runs, config, elapsed_seconds=9.5)
    loaded_runs, elapsed = _load_checkpoint(str(checkpoint), config)
    assert loaded_runs == runs
    assert elapsed == 9.5

    mismatched = dict(config)
    mismatched["seed"] = 999
    loaded_runs_mismatch, elapsed_mismatch = _load_checkpoint(
        str(checkpoint), mismatched
    )
    assert loaded_runs_mismatch == []
    assert elapsed_mismatch == 0.0


def test_build_provenance_includes_versions_and_seed():
    provenance = _build_provenance(seed=777)
    assert provenance["seed"] == 777
    assert provenance["python_version"]
    assert provenance["torch_version"]
    assert provenance["numpy_version"]
    assert provenance["scipy_version"]
    assert provenance["matplotlib_version"]
