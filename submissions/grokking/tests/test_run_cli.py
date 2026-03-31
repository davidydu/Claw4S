"""Tests for run.py CLI parsing and metadata helpers."""

from run import build_metadata, parse_csv_floats, parse_csv_ints


def test_parse_csv_floats():
    assert parse_csv_floats("0,0.001,0.1,1") == [0.0, 0.001, 0.1, 1.0]


def test_parse_csv_ints():
    assert parse_csv_ints("16,32,64") == [16, 32, 64]


def test_build_metadata_includes_grid_and_runtime():
    metadata = build_metadata(
        sweep_config={
            "weight_decays": [0.0, 0.1],
            "dataset_fractions": [0.5, 0.9],
            "hidden_dims": [16],
            "p": 97,
            "embed_dim": 16,
            "max_epochs": 2500,
            "seed": 42,
        },
        runtime_seconds=12.34,
        total_runs=4,
        phase_summary={
            "phase_counts": {
                "confusion": 1,
                "memorization": 1,
                "grokking": 1,
                "comprehension": 1,
            }
        },
        torch_version="2.6.0",
        numpy_version="2.2.4",
    )
    assert metadata["runtime_seconds"] == 12.34
    assert metadata["expected_total_runs"] == 4
    assert metadata["sweep"]["weight_decays"] == [0.0, 0.1]
    assert metadata["phase_counts"]["grokking"] == 1
    assert metadata["environment"]["torch_version"] == "2.6.0"
