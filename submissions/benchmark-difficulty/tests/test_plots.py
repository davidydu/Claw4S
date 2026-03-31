"""Tests for plotting module."""

import os
import pytest
from src.analysis import run_full_analysis
from src.plots import (
    plot_feature_correlations,
    plot_difficulty_prediction,
    plot_feature_importance,
)


@pytest.fixture
def results():
    """Run analysis on hardcoded data for plot tests."""
    return run_full_analysis(use_hardcoded=True, seed=42)


@pytest.fixture
def fig_dir(tmp_path):
    """Create a temporary figure directory."""
    d = tmp_path / "figures"
    d.mkdir()
    return str(d)


def test_plot_feature_correlations(results, fig_dir):
    """plot_feature_correlations creates a PNG file."""
    path = os.path.join(fig_dir, "feature_correlations.png")
    plot_feature_correlations(results["correlations"], path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000  # non-trivial file


def test_plot_difficulty_prediction(results, fig_dir):
    """plot_difficulty_prediction creates a PNG file."""
    path = os.path.join(fig_dir, "difficulty_prediction.png")
    plot_difficulty_prediction(
        results["predictions"], results["difficulties"], path
    )
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000


def test_plot_feature_importance(results, fig_dir):
    """plot_feature_importance creates a PNG file."""
    path = os.path.join(fig_dir, "feature_importance.png")
    plot_feature_importance(results["feature_importances"], path)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 1000
