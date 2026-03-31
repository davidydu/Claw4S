# tests/test_plots.py
"""Tests for plotting functions."""

import os
import tempfile
import numpy as np
from src.plots import (
    plot_zipf_fit,
    plot_piecewise_comparison,
    plot_alpha_compression_correlation,
)


def test_plot_zipf_fit_creates_file():
    """plot_zipf_fit must create a PNG file."""
    ranks = np.arange(1, 101)
    freqs = (1000 / ranks).astype(int)
    freqs[freqs == 0] = 1
    fit_params = {"alpha": 1.0, "q": 0.0, "C": 1000.0, "r_squared": 0.99}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_zipf.png")
        plot_zipf_fit(ranks, freqs, fit_params, "Test Zipf", path)
        assert os.path.exists(path), f"File not created: {path}"
        assert os.path.getsize(path) > 0


def test_plot_piecewise_comparison_creates_file():
    """plot_piecewise_comparison must create a PNG file."""
    results = [
        {
            "label": "English (gpt4o)",
            "piecewise_fit": {
                "head": {"alpha": 0.8, "r_squared": 0.95},
                "body": {"alpha": 1.1, "r_squared": 0.98},
                "tail": {"alpha": 1.5, "r_squared": 0.90},
            },
        },
        {
            "label": "Python (gpt4o)",
            "piecewise_fit": {
                "head": {"alpha": 0.6, "r_squared": 0.92},
                "body": {"alpha": 0.9, "r_squared": 0.96},
                "tail": {"alpha": 1.2, "r_squared": 0.88},
            },
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_piecewise.png")
        plot_piecewise_comparison(results, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0


def test_plot_alpha_compression_creates_file():
    """plot_alpha_compression_correlation must create a PNG file."""
    alphas = [0.8, 0.9, 1.0, 1.1, 1.2]
    compressions = [3.0, 3.5, 4.0, 4.5, 5.0]
    labels = ["a", "b", "c", "d", "e"]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_corr.png")
        plot_alpha_compression_correlation(alphas, compressions, labels, path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
