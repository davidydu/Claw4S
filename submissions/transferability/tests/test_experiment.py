"""Tests for experiment orchestration."""

import numpy as np
from src.experiment import compute_summary


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
