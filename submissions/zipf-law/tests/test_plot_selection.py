# tests/test_plot_selection.py
"""Tests for choosing representative Zipf fit figures."""

import numpy as np

from src.plot_selection import (
    build_zipf_fit_plot_plan,
    select_representative_analyses,
)


def test_select_representative_analyses_balances_corpus_types():
    """Representative cases should cover both natural language and code."""
    analyses = [
        {"label": "English (gpt4o)", "corpus": "English", "corpus_type": "natural_language"},
        {"label": "German (gpt4)", "corpus": "German", "corpus_type": "natural_language"},
        {"label": "French (mistral)", "corpus": "French", "corpus_type": "natural_language"},
        {"label": "Python code (gpt4o)", "corpus": "Python code", "corpus_type": "code"},
        {"label": "Java code (gpt4)", "corpus": "Java code", "corpus_type": "code"},
        {"label": "Python code (mistral)", "corpus": "Python code", "corpus_type": "code"},
    ]

    selected = select_representative_analyses(analyses, max_cases=4)

    assert [item["label"] for item in selected] == [
        "English (gpt4o)",
        "German (gpt4)",
        "Python code (gpt4o)",
        "Java code (gpt4)",
    ]


def test_build_zipf_fit_plot_plan_uses_exact_analysis_labels():
    """Plot planning should use stored data keyed by the exact analysis label."""
    analyses = [
        {
            "label": "English (gpt4o)",
            "corpus_type": "natural_language",
            "global_fit": {"alpha": 0.8, "q": 0.0, "C": 100.0, "r_squared": 0.95},
        },
        {
            "label": "Python code (gpt4o)",
            "corpus_type": "code",
            "global_fit": {"alpha": 1.2, "q": 0.0, "C": 80.0, "r_squared": 0.97},
        },
    ]
    plot_data_by_label = {
        "English (gpt4o)": {
            "ranks": np.array([1, 2, 3]),
            "freqs": np.array([10, 5, 3]),
        },
        "Python code (gpt4o)": {
            "ranks": np.array([1, 2, 3]),
            "freqs": np.array([12, 6, 4]),
        },
    }

    plan = build_zipf_fit_plot_plan(analyses, plot_data_by_label, max_cases=4)

    assert [item["label"] for item in plan] == [
        "English (gpt4o)",
        "Python code (gpt4o)",
    ]
    assert all("ranks" in item and "freqs" in item for item in plan)


def test_select_representative_analyses_prefers_unique_corpora():
    """Representative cases should avoid repeating the same corpus by tokenizer."""
    analyses = [
        {
            "label": "English (gpt4o)",
            "corpus": "English",
            "corpus_type": "natural_language",
        },
        {
            "label": "English (gpt4)",
            "corpus": "English",
            "corpus_type": "natural_language",
        },
        {
            "label": "German (gpt4o)",
            "corpus": "German",
            "corpus_type": "natural_language",
        },
        {
            "label": "Python code (gpt4o)",
            "corpus": "Python code",
            "corpus_type": "code",
        },
        {
            "label": "Python code (gpt4)",
            "corpus": "Python code",
            "corpus_type": "code",
        },
        {
            "label": "Java code (gpt4o)",
            "corpus": "Java code",
            "corpus_type": "code",
        },
    ]

    selected = select_representative_analyses(analyses, max_cases=4)

    assert [item["label"] for item in selected] == [
        "English (gpt4o)",
        "German (gpt4o)",
        "Python code (gpt4o)",
        "Java code (gpt4o)",
    ]
