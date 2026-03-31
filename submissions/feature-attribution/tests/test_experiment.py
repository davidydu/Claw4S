"""Tests for experiment orchestration and reporting."""

import numpy as np
import torch

import src.experiment as experiment


def _minimal_results(summary):
    """Build a minimal results payload for report generation tests."""
    return {
        "metadata": {
            "depths": [1, 4],
            "width": 64,
            "n_samples": 500,
            "n_features": 10,
            "n_classes": 5,
            "n_test": 100,
            "seeds": [42, 123, 456],
            "n_steps": 50,
            "elapsed_seconds": 6.3,
        },
        "per_depth": {
            "1": {
                "accuracy_mean": 1.0,
                "accuracy_std": 0.0,
                "agreement": {
                    "vanilla_gradient_vs_gradient_x_input": {"mean": 0.719, "std": 0.194},
                    "gradient_x_input_vs_integrated_gradients": {"mean": 0.950, "std": 0.055},
                },
            },
            "4": {
                "accuracy_mean": 1.0,
                "accuracy_std": 0.0,
                "agreement": {
                    "vanilla_gradient_vs_gradient_x_input": {"mean": 0.780, "std": 0.144},
                    "gradient_x_input_vs_integrated_gradients": {"mean": 0.937, "std": 0.064},
                },
            },
        },
        "summary": summary,
    }


def test_run_experiment_targets_predicted_class(monkeypatch, tmp_path):
    """Attributions should be computed for the model's predicted class."""
    captured_targets = []

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def __call__(self, x):
            logits = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
            return logits.repeat(x.shape[0], 1)

    def fake_make_gaussian_clusters(**kwargs):
        X = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y = np.array([1, 0, 0], dtype=np.int64)
        return X, y

    def fake_train_model(model, X_train, y_train, **kwargs):
        return [0.5]

    def fake_compute_all_attributions(model, x, target_class, n_steps):
        captured_targets.append(target_class)
        return {
            "vanilla_gradient": np.array([1.0, 2.0]),
            "gradient_x_input": np.array([1.0, 2.0]),
            "integrated_gradients": np.array([1.0, 2.0]),
        }

    monkeypatch.setattr(experiment, "MLP", FakeModel)
    monkeypatch.setattr(experiment, "make_gaussian_clusters", fake_make_gaussian_clusters)
    monkeypatch.setattr(experiment, "train_model", fake_train_model)
    monkeypatch.setattr(experiment, "compute_all_attributions", fake_compute_all_attributions)

    experiment.run_experiment(
        depths=[1],
        n_samples=3,
        n_features=2,
        n_classes=2,
        n_test=1,
        seeds=[7],
        epochs=1,
        results_dir=str(tmp_path),
    )

    assert captured_targets == [0]


def test_generate_report_uses_actual_train_test_split():
    """Report should distinguish total samples from the train/test split."""
    summary = {
        "overall_mean_rho": 0.85,
        "substantial_disagreement": False,
        "pair_trends": {
            "vanilla_gradient_vs_gradient_x_input": {
                "overall_mean_rho": 0.75,
                "depth_trend": 0.061,
                "per_depth_means": [0.719, 0.780],
            },
            "gradient_x_input_vs_integrated_gradients": {
                "overall_mean_rho": 0.944,
                "depth_trend": -0.013,
                "per_depth_means": [0.950, 0.937],
            },
        },
    }

    report = experiment._generate_report(_minimal_results(summary))

    assert "- Samples: 500 total (400 train / 100 test)" in report


def test_generate_report_describes_mixed_depth_trends():
    """Report should avoid a one-direction depth claim when trends are mixed."""
    summary = {
        "overall_mean_rho": 0.85,
        "substantial_disagreement": False,
        "pair_trends": {
            "vanilla_gradient_vs_gradient_x_input": {
                "overall_mean_rho": 0.75,
                "depth_trend": 0.061,
                "per_depth_means": [0.719, 0.780],
            },
            "gradient_x_input_vs_integrated_gradients": {
                "overall_mean_rho": 0.944,
                "depth_trend": -0.013,
                "per_depth_means": [0.950, 0.937],
            },
        },
    }

    report = experiment._generate_report(_minimal_results(summary))

    assert "Depth effects are mixed across method pairs in this configuration." in report


def test_run_experiment_writes_stable_artifacts_across_reruns(monkeypatch, tmp_path):
    """Saved result artifacts should not change solely because runtime changed."""

    class FakeModel:
        def __init__(self, *args, **kwargs):
            pass

        def eval(self):
            return self

        def __call__(self, x):
            logits = torch.tensor([[2.0, 1.0]], dtype=torch.float32)
            return logits.repeat(x.shape[0], 1)

    def fake_make_gaussian_clusters(**kwargs):
        X = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        y = np.array([0, 0, 0], dtype=np.int64)
        return X, y

    def fake_train_model(model, X_train, y_train, **kwargs):
        return [0.5]

    def fake_compute_all_attributions(model, x, target_class, n_steps):
        return {
            "vanilla_gradient": np.array([1.0, 2.0]),
            "gradient_x_input": np.array([1.0, 2.0]),
            "integrated_gradients": np.array([1.0, 2.0]),
        }

    time_values = iter([100.0, 106.2, 200.0, 208.8])

    monkeypatch.setattr(experiment, "MLP", FakeModel)
    monkeypatch.setattr(experiment, "make_gaussian_clusters", fake_make_gaussian_clusters)
    monkeypatch.setattr(experiment, "train_model", fake_train_model)
    monkeypatch.setattr(experiment, "compute_all_attributions", fake_compute_all_attributions)
    monkeypatch.setattr(experiment.time, "time", lambda: next(time_values))

    experiment.run_experiment(
        depths=[1],
        n_samples=3,
        n_features=2,
        n_classes=2,
        n_test=1,
        seeds=[7],
        epochs=1,
        results_dir=str(tmp_path),
    )
    first_results = (tmp_path / "results.json").read_text()
    first_report = (tmp_path / "report.md").read_text()

    experiment.run_experiment(
        depths=[1],
        n_samples=3,
        n_features=2,
        n_classes=2,
        n_test=1,
        seeds=[7],
        epochs=1,
        results_dir=str(tmp_path),
    )
    second_results = (tmp_path / "results.json").read_text()
    second_report = (tmp_path / "report.md").read_text()

    assert first_results == second_results
    assert first_report == second_report
