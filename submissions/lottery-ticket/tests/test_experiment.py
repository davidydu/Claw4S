"""Tests for experiment-level reproducibility metadata."""

from src.experiment import collect_runtime_metadata


def test_collect_runtime_metadata_includes_reproducibility_fields():
    """Runtime metadata should record deterministic and environment details."""
    meta = collect_runtime_metadata(elapsed_seconds=12.3, total_runs=99)

    assert meta["total_runs"] == 99
    assert meta["elapsed_seconds"] == 12.3
    assert "python_version" in meta and meta["python_version"]
    assert "torch_version" in meta and meta["torch_version"]
    assert "numpy_version" in meta and meta["numpy_version"]
    assert "platform" in meta and meta["platform"]
    assert "deterministic_algorithms_enabled" in meta
    assert meta["seed_values"] == [42, 123, 7]
