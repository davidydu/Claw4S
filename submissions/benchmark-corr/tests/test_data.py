"""Tests for src/data.py — verify hardcoded benchmark data integrity."""

import numpy as np
from src.data import (
    BENCHMARKS, MODEL_INFO, SCORES,
    get_model_names, get_model_families, get_model_params,
    get_scores_dataframe, get_family_indices, get_data_fingerprint,
)


def test_scores_shape():
    """SCORES shape matches MODEL_INFO x BENCHMARKS."""
    assert SCORES.shape == (len(MODEL_INFO), len(BENCHMARKS))


def test_at_least_30_models():
    """At least 30 models are hardcoded."""
    assert len(MODEL_INFO) >= 30


def test_at_least_6_benchmarks():
    """At least 6 benchmarks are hardcoded."""
    assert len(BENCHMARKS) >= 6


def test_scores_in_valid_range():
    """All scores are between 0 and 100 (percentage accuracy)."""
    assert np.all(SCORES >= 0.0)
    assert np.all(SCORES <= 100.0)


def test_no_nan_scores():
    """No NaN values in scores."""
    assert not np.any(np.isnan(SCORES))


def test_model_names_unique():
    """All model names are unique."""
    names = get_model_names()
    assert len(names) == len(set(names))


def test_model_params_positive():
    """All parameter counts are positive."""
    params = get_model_params()
    assert np.all(params > 0)


def test_multiple_families():
    """At least 5 different model families."""
    families = set(get_model_families())
    assert len(families) >= 5


def test_get_scores_dataframe():
    """get_scores_dataframe returns correct structure."""
    df = get_scores_dataframe()
    assert len(df) == len(MODEL_INFO)
    first_model = get_model_names()[0]
    assert set(df[first_model].keys()) == set(BENCHMARKS)


def test_get_family_indices():
    """get_family_indices maps families to valid indices."""
    fi = get_family_indices()
    assert len(fi) >= 5
    for fam, indices in fi.items():
        for idx in indices:
            assert 0 <= idx < len(MODEL_INFO)
            assert MODEL_INFO[idx][1] == fam


def test_llama2_scores_reasonable():
    """Spot-check: Llama-2-70B should score higher than Llama-2-7B on MMLU."""
    names = get_model_names()
    i7b = names.index("Llama-2-7B")
    i70b = names.index("Llama-2-70B")
    mmlu_col = BENCHMARKS.index("MMLU")
    assert SCORES[i70b, mmlu_col] > SCORES[i7b, mmlu_col]


def test_small_models_lower_scores():
    """Small models (< 0.5B) should generally score lower than large models (> 10B)."""
    params = get_model_params()
    small_mask = params < 0.5
    large_mask = params > 10.0
    # Compare average scores across all benchmarks
    avg_small = SCORES[small_mask].mean()
    avg_large = SCORES[large_mask].mean()
    assert avg_large > avg_small


def test_data_fingerprint_is_stable_sha256():
    """Data fingerprint should be deterministic and look like a SHA-256 hex digest."""
    fp1 = get_data_fingerprint()
    fp2 = get_data_fingerprint()
    assert fp1 == fp2
    assert len(fp1) == 64
    assert all(ch in "0123456789abcdef" for ch in fp1)
