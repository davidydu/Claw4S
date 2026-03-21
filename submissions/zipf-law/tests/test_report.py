# tests/test_report.py
"""Tests for report generation."""

import os
import tempfile
from src.report import generate_report


def _make_sample_results():
    """Create minimal results dict for testing."""
    return {
        "metadata": {
            "timestamp": "2026-03-21T00:00:00Z",
            "num_tokenizers": 2,
            "num_corpora": 3,
            "seed": 42,
        },
        "analyses": [
            {
                "tokenizer": "gpt4o",
                "corpus": "English",
                "corpus_type": "natural_language",
                "compression_ratio": 4.5,
                "global_fit": {
                    "alpha": 1.05,
                    "q": 0.0,
                    "r_squared": 0.97,
                    "C": 500.0,
                },
                "piecewise_fit": {
                    "head": {"alpha": 0.8, "r_squared": 0.95},
                    "body": {"alpha": 1.1, "r_squared": 0.98},
                    "tail": {"alpha": 1.5, "r_squared": 0.90},
                },
                "breakpoints": [50, 200],
                "num_total_tokens": 5000,
                "num_unique_tokens": 800,
            },
            {
                "tokenizer": "gpt4o",
                "corpus": "Python",
                "corpus_type": "code",
                "compression_ratio": 3.2,
                "global_fit": {
                    "alpha": 0.85,
                    "q": 1.0,
                    "r_squared": 0.93,
                    "C": 400.0,
                },
                "piecewise_fit": {
                    "head": {"alpha": 0.6, "r_squared": 0.92},
                    "body": {"alpha": 0.9, "r_squared": 0.96},
                    "tail": {"alpha": 1.2, "r_squared": 0.85},
                },
                "breakpoints": [30],
                "num_total_tokens": 4000,
                "num_unique_tokens": 600,
            },
        ],
        "correlation": {
            "pearson_r": 0.75,
            "pearson_p": 0.01,
            "spearman_r": 0.72,
            "spearman_p": 0.02,
        },
    }


def test_generate_report_returns_nonempty_string():
    """Report must be a non-empty string."""
    results = _make_sample_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.md")
        report = generate_report(results, output_path=path)
        assert isinstance(report, str)
        assert len(report) > 100


def test_generate_report_has_required_sections():
    """Report must contain key section headers."""
    results = _make_sample_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.md")
        report = generate_report(results, output_path=path)
        assert "# Zipf" in report
        assert "Global" in report or "alpha" in report.lower()
        assert "Piecewise" in report or "Region" in report
        assert "Correlation" in report or "correlation" in report


def test_generate_report_writes_file():
    """Report must be written to disk."""
    results = _make_sample_results()
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "report.md")
        generate_report(results, output_path=path)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
