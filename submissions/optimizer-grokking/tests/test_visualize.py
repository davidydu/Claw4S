"""Tests for report generation and uncertainty summaries."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from visualize import generate_report, wilson_interval


def _make_run(
    optimizer: str,
    lr: float,
    weight_decay: float,
    outcome: str,
    train_acc: float = 0.99,
    test_acc: float = 0.98,
) -> dict:
    return {
        "optimizer": optimizer,
        "lr": lr,
        "weight_decay": weight_decay,
        "outcome": outcome,
        "final_train_acc": train_acc,
        "final_test_acc": test_acc,
        "memorization_epoch": 75 if outcome in {"grokking", "direct_generalization", "memorization"} else None,
        "generalization_epoch": 150 if outcome == "grokking" else (75 if outcome == "direct_generalization" else None),
        "grokking_epoch": 150 if outcome == "grokking" else None,
        "history": [{
            "epoch": 1,
            "train_acc": 0.1,
            "test_acc": 0.1,
            "train_loss": 1.0,
            "test_loss": 1.0,
        }],
    }


def test_wilson_interval_zero_successes():
    """Wilson interval upper bound should be non-zero even for 0/n."""
    lower, upper = wilson_interval(0, 9)
    assert lower == 0.0
    assert 0.29 < upper < 0.31


def test_generate_report_includes_uncertainty_section(tmp_path):
    """Report should include Wilson confidence intervals for grokking rates."""
    data = {
        "metadata": {
            "prime": 97,
            "seed": 42,
            "max_epochs": 10,
            "batch_size": 32,
            "num_runs": 4,
            "total_seconds": 1.0,
            "optimizers": ["sgd", "sgd_momentum", "adam", "adamw"],
            "learning_rates": [0.03],
            "weight_decays": [0.0],
            "python_version": "3.13.5",
            "torch_version": "2.6.0",
            "numpy_version": "2.2.4",
            "platform": "test-platform",
        },
        "runs": [
            _make_run("sgd", 0.03, 0.0, "failure", train_acc=0.2, test_acc=0.05),
            _make_run("sgd_momentum", 0.03, 0.0, "memorization", train_acc=1.0, test_acc=0.4),
            _make_run("adam", 0.03, 0.0, "grokking"),
            _make_run("adamw", 0.03, 0.0, "direct_generalization"),
        ],
    }

    report = generate_report(data, results_dir=str(tmp_path))
    assert "## Statistical Uncertainty" in report
    assert "Wilson 95% CI" in report
    assert "| adam | 1/1 | 100.0% |" in report
