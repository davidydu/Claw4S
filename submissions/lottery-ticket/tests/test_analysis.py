"""Tests for analysis helpers and report-ready summary outputs."""

import csv

from src.analysis import compute_summary_stats, export_summary_csv


def test_compute_summary_stats_includes_confidence_intervals():
    """Summary stats should include 95% CI bounds for each configuration."""
    results = [
        {
            "task": "modular",
            "strategy": "magnitude",
            "sparsity": 0.7,
            "test_acc": 0.20,
            "epochs_trained": 100,
        },
        {
            "task": "modular",
            "strategy": "magnitude",
            "sparsity": 0.7,
            "test_acc": 0.35,
            "epochs_trained": 110,
        },
        {
            "task": "modular",
            "strategy": "magnitude",
            "sparsity": 0.7,
            "test_acc": 0.40,
            "epochs_trained": 120,
        },
    ]

    summary = compute_summary_stats(results)
    cell = summary["modular"]["magnitude"][0.7]

    assert "metric_ci_low" in cell
    assert "metric_ci_high" in cell
    assert cell["metric_ci_low"] <= cell["metric_mean"] <= cell["metric_ci_high"]
    assert cell["n_seeds"] == 3


def test_export_summary_csv_writes_expected_columns(tmp_path):
    """Summary CSV should include reproducible, machine-readable analysis fields."""
    results = [
        {
            "task": "modular",
            "strategy": "structured",
            "sparsity": 0.5,
            "test_acc": 0.60,
            "epochs_trained": 200,
        },
        {
            "task": "modular",
            "strategy": "structured",
            "sparsity": 0.5,
            "test_acc": 0.65,
            "epochs_trained": 210,
        },
        {
            "task": "modular",
            "strategy": "structured",
            "sparsity": 0.5,
            "test_acc": 0.70,
            "epochs_trained": 220,
        },
    ]
    summary = compute_summary_stats(results)

    csv_path = export_summary_csv(summary, output_dir=str(tmp_path))
    assert csv_path.endswith("summary.csv")

    with open(csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    row = rows[0]
    assert row["task"] == "modular"
    assert row["strategy"] == "structured"
    assert row["metric_name"] == "test_acc"
    assert row["metric_ci_low"] != ""
    assert row["metric_ci_high"] != ""
