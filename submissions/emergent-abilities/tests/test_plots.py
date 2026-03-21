"""Tests for src/plots.py -- visualization functions."""

import os
import tempfile

from src.analysis import (
    compute_metric_comparison,
    compute_nonlinearity_scores,
    generate_synthetic_demo,
    compute_mmlu_analysis,
)
from src.plots import (
    plot_metric_comparison,
    plot_synthetic_demo,
    plot_nonlinearity_heatmap,
    plot_mmlu_scaling,
)


def test_plot_metric_comparison_creates_file():
    """Metric comparison plot creates a PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "metric_comparison.png")
        comparison = compute_metric_comparison("2_digit_multiplication")
        plot_metric_comparison(comparison, outpath)
        assert os.path.exists(outpath), f"File not created: {outpath}"


def test_plot_synthetic_demo_creates_file():
    """Synthetic demo plot creates a PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "synthetic_demo.png")
        demo = generate_synthetic_demo(seed=42)
        plot_synthetic_demo(demo, outpath)
        assert os.path.exists(outpath), f"File not created: {outpath}"


def test_plot_nonlinearity_heatmap_creates_file():
    """Nonlinearity heatmap creates a PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "nonlinearity_heatmap.png")
        scores = compute_nonlinearity_scores()
        plot_nonlinearity_heatmap(scores, outpath)
        assert os.path.exists(outpath), f"File not created: {outpath}"


def test_plot_mmlu_scaling_creates_file():
    """MMLU scaling plot creates a PNG file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        outpath = os.path.join(tmpdir, "mmlu_scaling.png")
        mmlu = compute_mmlu_analysis()
        plot_mmlu_scaling(mmlu, outpath)
        assert os.path.exists(outpath), f"File not created: {outpath}"


def test_plots_are_png():
    """All plot files have correct PNG headers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        demo = generate_synthetic_demo(seed=42)
        outpath = os.path.join(tmpdir, "test.png")
        plot_synthetic_demo(demo, outpath)

        with open(outpath, "rb") as f:
            header = f.read(8)
        # PNG magic bytes
        assert header[:4] == b"\x89PNG", "File does not have PNG header"
