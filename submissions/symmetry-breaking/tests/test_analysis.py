"""Tests for analysis and summary functions."""

import pytest

from src.analysis import compute_breaking_epoch, generate_report, summarize_results


def test_breaking_epoch_found():
    """Should find the correct breaking epoch."""
    epochs = [0, 50, 100, 150, 200]
    symmetry = [1.0, 0.8, 0.4, 0.2, 0.1]
    assert compute_breaking_epoch(epochs, symmetry, threshold=0.5) == 100


def test_breaking_epoch_never():
    """Should return -1 if symmetry never drops below threshold."""
    epochs = [0, 50, 100]
    symmetry = [1.0, 0.9, 0.8]
    assert compute_breaking_epoch(epochs, symmetry, threshold=0.5) == -1


def test_summarize_results():
    """Summary should contain expected keys."""
    fake_results = [
        {
            "hidden_dim": 16,
            "epsilon": 0.0,
            "seed": 42,
            "epochs_logged": [0, 50],
            "symmetry_values": [1.0, 0.95],
            "loss_values": [2.0],
            "train_acc_values": [0.5],
            "final_test_acc": 0.1,
            "final_train_acc": 0.15,
            "initial_symmetry": 1.0,
            "final_symmetry": 0.95,
            "num_epochs": 50,
            "batch_size": 256,
            "lr": 0.1,
            "modulus": 97,
        },
        {
            "hidden_dim": 16,
            "epsilon": 0.01,
            "seed": 42,
            "epochs_logged": [0, 50],
            "symmetry_values": [0.99, 0.3],
            "loss_values": [2.0],
            "train_acc_values": [0.5],
            "final_test_acc": 0.4,
            "final_train_acc": 0.5,
            "initial_symmetry": 0.99,
            "final_symmetry": 0.3,
            "num_epochs": 50,
            "batch_size": 256,
            "lr": 0.1,
            "modulus": 97,
        },
    ]
    summary = summarize_results(fake_results)
    assert summary["num_runs"] == 2
    assert "zero_eps_final_symmetry_mean" in summary
    assert "nonzero_eps_final_symmetry_mean" in summary
    assert len(summary["runs"]) == 2
    assert summary["chance_accuracy"] == pytest.approx(1.0 / 97.0)
    assert summary["best_test_acc"] == pytest.approx(0.4)
    assert summary["best_run_hidden_dim"] == 16
    assert summary["best_run_epsilon"] == pytest.approx(0.01)
    assert summary["best_test_acc_at_min_epsilon"] == pytest.approx(0.1)
    assert summary["best_test_acc_at_max_epsilon"] == pytest.approx(0.4)
    assert summary["accuracy_gain_max_vs_min_epsilon"] == pytest.approx(0.3)


def test_generate_report_includes_methodological_note():
    """Report should clarify that only the incoming layer is symmetrized."""
    fake_results = [
        {
            "hidden_dim": 16,
            "epsilon": 0.0,
            "seed": 42,
            "epochs_logged": [0, 50],
            "symmetry_values": [1.0, 0.95],
            "loss_values": [2.0],
            "train_acc_values": [0.5],
            "final_test_acc": 0.1,
            "final_train_acc": 0.15,
            "initial_symmetry": 1.0,
            "final_symmetry": 0.95,
            "num_epochs": 50,
            "batch_size": 256,
            "lr": 0.1,
            "modulus": 97,
        }
    ]
    summary = summarize_results(fake_results)

    report = generate_report(fake_results, summary)

    assert "## Methodological Note" in report
    assert "`W1`" in report
    assert "`W2`" in report
    assert "Chance-level accuracy" in report
