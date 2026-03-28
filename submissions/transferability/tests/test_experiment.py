"""Tests for experiment orchestration."""

import numpy as np
import torch
from src.experiment import compute_summary
from src import experiment
from torch.utils.data import TensorDataset


def test_compute_summary_structure():
    """compute_summary returns all expected keys."""
    same_arch = [
        {
            "source_width": 32, "target_width": 32, "capacity_ratio": 1.0,
            "transfer_rate": 0.8, "source_clean_acc": 0.9,
        },
        {
            "source_width": 32, "target_width": 64, "capacity_ratio": 2.0,
            "transfer_rate": 0.5, "source_clean_acc": 0.9,
        },
    ]
    cross_depth = [
        {
            "source_width": 32, "target_width": 32, "capacity_ratio": 1.0,
            "transfer_rate": 0.6, "source_clean_acc": 0.9,
        },
    ]

    summary = compute_summary(same_arch, cross_depth)
    assert "diagonal_mean_transfer" in summary
    assert "off_diagonal_mean_transfer" in summary
    assert "transfer_by_capacity_ratio" in summary
    assert "same_width_same_depth_mean" in summary
    assert "same_width_cross_depth_mean" in summary
    assert "n_same_arch_runs" in summary
    assert "n_cross_depth_runs" in summary


def test_compute_summary_values():
    """compute_summary produces correct aggregate values."""
    same_arch = [
        {"source_width": 32, "target_width": 32, "capacity_ratio": 1.0, "transfer_rate": 0.8},
        {"source_width": 32, "target_width": 32, "capacity_ratio": 1.0, "transfer_rate": 0.6},
        {"source_width": 32, "target_width": 64, "capacity_ratio": 2.0, "transfer_rate": 0.4},
    ]
    cross_depth = []

    summary = compute_summary(same_arch, cross_depth)
    assert summary["diagonal_mean_transfer"] == round(np.mean([0.8, 0.6]), 4)
    assert summary["off_diagonal_mean_transfer"] == round(np.mean([0.4]), 4)
    assert summary["n_same_arch_runs"] == 3
    assert summary["n_cross_depth_runs"] == 0


def test_cross_depth_reuses_trained_models(monkeypatch, tmp_path):
    """run_cross_depth_experiment trains each source/target model once per seed."""
    trained_models = []

    def fake_make_gaussian_clusters(
        n_samples: int,
        n_features: int,
        n_classes: int,
        seed: int,
    ) -> TensorDataset:
        X = torch.zeros((n_samples, n_features), dtype=torch.float32)
        y = torch.zeros(n_samples, dtype=torch.int64)
        return TensorDataset(X, y)

    def fake_train_model(model, dataset, lr, epochs, batch_size, seed):
        trained_models.append((seed, model.hidden_width, model.n_hidden_layers))
        return {"final_loss": 0.0, "final_accuracy": 1.0}

    def fake_compute_transfer_rate(source_model, target_model, X, y, epsilon):
        return {
            "transfer_rate": 0.5,
            "source_clean_acc": 1.0,
            "target_clean_acc": 1.0,
            "source_adv_acc": 0.0,
            "target_adv_acc": 0.0,
            "n_successful_source_advs": 1,
        }

    monkeypatch.setattr(experiment, "make_gaussian_clusters", fake_make_gaussian_clusters)
    monkeypatch.setattr(experiment, "train_model", fake_train_model)
    monkeypatch.setattr(experiment, "compute_transfer_rate", fake_compute_transfer_rate)

    results = experiment.run_cross_depth_experiment(tmp_path)

    assert len(results) == len(experiment.SEEDS) * len(experiment.WIDTHS) ** 2
    assert len(trained_models) == len(experiment.SEEDS) * len(experiment.WIDTHS) * 2
    assert len(set(trained_models)) == len(trained_models)
