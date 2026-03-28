"""Tests for validation helper checks."""

from validate import (
    build_required_files,
    find_grid_coverage_errors,
    find_phase_consistency_errors,
    find_phase_summary_errors,
)


def _result(
    hidden_dim: int,
    weight_decay: float,
    train_fraction: float,
    phase: str,
    final_train_acc: float = 0.99,
    final_test_acc: float = 0.99,
    epoch_train_95: int | None = 100,
    epoch_test_95: int | None = 400,
    grokking_gap: int | None = 300,
) -> dict:
    return {
        "config": {
            "hidden_dim": hidden_dim,
            "weight_decay": weight_decay,
            "train_fraction": train_fraction,
            "param_count": 1000,
            "seed": 42,
        },
        "metrics": {
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
            "epoch_train_95": epoch_train_95,
            "epoch_test_95": epoch_test_95,
            "total_epochs": 500,
            "train_accs": [0.9, final_train_acc],
            "test_accs": [0.4, final_test_acc],
            "train_losses": [1.0, 0.1],
            "test_losses": [1.2, 0.2],
            "logged_epochs": [100, 500],
        },
        "phase": phase,
        "grokking_gap": grokking_gap,
        "elapsed_seconds": 1.0,
    }


def test_build_required_files_uses_hidden_dims():
    files = build_required_files([16, 64])
    assert "results/sweep_results.json" in files
    assert "results/phase_diagram_h16.png" in files
    assert "results/phase_diagram_h64.png" in files


def test_find_grid_coverage_errors_detects_missing_combo():
    sweep = [
        _result(16, 0.0, 0.3, "grokking"),
        _result(16, 0.0, 0.5, "grokking"),
        _result(16, 0.1, 0.3, "grokking"),
    ]
    errors = find_grid_coverage_errors(sweep)
    assert any("Missing grid point" in e for e in errors)


def test_find_grid_coverage_errors_detects_duplicate_combo():
    sweep = [
        _result(16, 0.0, 0.3, "grokking"),
        _result(16, 0.0, 0.3, "grokking"),
        _result(16, 0.0, 0.5, "grokking"),
        _result(16, 0.1, 0.3, "grokking"),
        _result(16, 0.1, 0.5, "grokking"),
    ]
    errors = find_grid_coverage_errors(sweep)
    assert any("Duplicate grid point" in e for e in errors)


def test_find_phase_consistency_errors_detects_mismatch():
    bad = _result(
        16,
        0.0,
        0.3,
        phase="grokking",
        final_train_acc=0.5,
        final_test_acc=0.2,
        epoch_train_95=None,
        epoch_test_95=None,
        grokking_gap=None,
    )
    errors = find_phase_consistency_errors([bad])
    assert any("Phase mismatch" in e for e in errors)


def test_find_phase_summary_errors_detects_incorrect_counts():
    sweep = [
        _result(16, 0.0, 0.3, "grokking"),
        _result(16, 0.0, 0.5, "memorization"),
    ]
    summary = {
        "phase_counts": {
            "confusion": 0,
            "memorization": 2,
            "grokking": 0,
            "comprehension": 0,
        },
        "total_runs": 2,
        "grokking_fraction": 0.0,
        "mean_grokking_gap": None,
        "max_grokking_gap": None,
    }
    errors = find_phase_summary_errors(sweep, summary)
    assert any("phase_counts mismatch" in e for e in errors)
