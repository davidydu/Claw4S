"""Tests for markdown report generation."""

from src.report import generate_report


def test_generate_report_handles_tied_best_depths_without_arbitrary_winner():
    """Tied top metrics should be reported as ties, not single winners."""
    results = {
        "metadata": {
            "seed": 42,
            "param_budgets": [5000],
            "depths": [1, 2],
            "n_bits": 20,
            "k_relevant": 3,
            "torch_version": "2.6.0",
            "num_experiments": 2,
            "task_hparams": {
                "sparse_parity": {
                    "lr": 3e-3,
                    "weight_decay": 1e-2,
                    "max_epochs": 1500,
                    "convergence_threshold": 0.85,
                }
            },
        },
        "results": [
            {
                "param_budget": 5000,
                "num_hidden_layers": 1,
                "hidden_width": 217,
                "actual_params": 4993,
                "task_name": "sparse_parity",
                "task_type": "classification",
                "metric_name": "accuracy",
                "best_test_metric": 1.0,
                "convergence_epoch": 343,
            },
            {
                "param_budget": 5000,
                "num_hidden_layers": 2,
                "hidden_width": 60,
                "actual_params": 5042,
                "task_name": "sparse_parity",
                "task_type": "classification",
                "metric_name": "accuracy",
                "best_test_metric": 1.0,
                "convergence_epoch": 79,
            },
        ],
    }

    report = generate_report(results)

    assert (
        "- **5K params**: tie among depths=1, 2 (Accuracy=1.0000); "
        "fastest among tied: depth=2 (epoch 79)"
    ) in report
    assert (
        "1. **sparse_parity**: Best overall Accuracy=1.0000, achieved by 2 "
        "configurations across depths 1, 2."
    ) in report


def test_generate_report_numbers_key_findings_sequentially():
    """Each task finding should advance the numbered list."""
    results = {
        "metadata": {
            "seed": 42,
            "param_budgets": [5000],
            "depths": [1, 2],
            "torch_version": "2.6.0",
            "num_experiments": 2,
            "task_hparams": {
                "smooth_regression": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "max_epochs": 800,
                    "convergence_threshold": 0.90,
                },
                "sparse_parity": {
                    "lr": 3e-3,
                    "weight_decay": 1e-2,
                    "max_epochs": 1500,
                    "convergence_threshold": 0.85,
                },
            },
        },
        "results": [
            {
                "param_budget": 5000,
                "num_hidden_layers": 2,
                "hidden_width": 65,
                "actual_params": 4941,
                "task_name": "smooth_regression",
                "task_type": "regression",
                "metric_name": "r_squared",
                "best_test_metric": 0.9240,
                "convergence_epoch": 273,
            },
            {
                "param_budget": 5000,
                "num_hidden_layers": 1,
                "hidden_width": 217,
                "actual_params": 4993,
                "task_name": "sparse_parity",
                "task_type": "classification",
                "metric_name": "accuracy",
                "best_test_metric": 1.0,
                "convergence_epoch": 343,
            },
        ],
    }

    report = generate_report(results)

    assert "1. **smooth_regression**" in report
    assert "2. **sparse_parity**" in report


def test_generate_report_includes_validation_selection_protocol():
    """Report metadata should document validation-based model selection."""
    results = {
        "metadata": {
            "seed": 42,
            "param_budgets": [5000],
            "depths": [1],
            "torch_version": "2.6.0",
            "num_experiments": 1,
            "model_selection": "validation_split",
            "validation_split_fraction": 0.2,
            "task_hparams": {
                "smooth_regression": {
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "max_epochs": 800,
                    "convergence_threshold": 0.90,
                }
            },
        },
        "results": [
            {
                "param_budget": 5000,
                "num_hidden_layers": 1,
                "hidden_width": 500,
                "actual_params": 5001,
                "task_name": "smooth_regression",
                "task_type": "regression",
                "metric_name": "r_squared",
                "best_test_metric": 0.90,
                "best_val_metric": 0.91,
                "convergence_epoch": 100,
            },
        ],
    }

    report = generate_report(results)

    assert (
        "- Model selection: validation split (20% of training data); "
        "test metric evaluated once at the best validation epoch"
    ) in report
    assert "### Validation Convergence Speed" in report
