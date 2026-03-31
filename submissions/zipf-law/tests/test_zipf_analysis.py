# tests/test_zipf_analysis.py
"""Tests for Zipf analysis functions."""

import numpy as np
from src.zipf_analysis import (
    compute_rank_frequency,
    fit_zipf_mandelbrot,
    fit_piecewise_zipf,
    detect_breakpoints,
    analyze_corpus,
)

# Seed for reproducibility
np.random.seed(42)


def _generate_zipfian_tokens(alpha: float, n: int, vocab_size: int) -> list[int]:
    """Generate synthetic token IDs following a Zipf distribution."""
    ranks = np.arange(1, vocab_size + 1)
    probs = ranks.astype(float) ** (-alpha)
    probs /= probs.sum()
    return list(np.random.choice(vocab_size, size=n, p=probs))


def test_compute_rank_frequency_sorted():
    """Frequencies must be sorted in descending order."""
    tokens = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3]
    ranks, freqs = compute_rank_frequency(tokens)
    assert len(ranks) == 4  # 4 unique tokens
    assert np.all(freqs[:-1] >= freqs[1:]), "Frequencies not sorted descending"


def test_compute_rank_frequency_ranks_start_at_one():
    """Ranks should start at 1."""
    tokens = [10, 20, 20, 30, 30, 30]
    ranks, freqs = compute_rank_frequency(tokens)
    assert ranks[0] == 1
    assert ranks[-1] == len(ranks)


def test_compute_rank_frequency_correct_counts():
    """Verify frequency counts match token occurrences."""
    tokens = [0, 0, 0, 1, 1, 2]
    ranks, freqs = compute_rank_frequency(tokens)
    assert freqs[0] == 3  # token 0 appears 3 times (rank 1)
    assert freqs[1] == 2  # token 1 appears 2 times (rank 2)
    assert freqs[2] == 1  # token 2 appears 1 time (rank 3)


def test_fit_zipf_mandelbrot_on_zipfian_data():
    """Fitting Zipfian data should recover alpha near input and high R^2."""
    tokens = _generate_zipfian_tokens(alpha=1.0, n=10000, vocab_size=500)
    ranks, freqs = compute_rank_frequency(tokens)
    result = fit_zipf_mandelbrot(ranks, freqs)
    assert abs(result["alpha"] - 1.0) < 0.3, f"alpha={result['alpha']}, expected ~1.0"
    assert result["r_squared"] > 0.90, f"R^2={result['r_squared']}, expected > 0.90"


def test_fit_zipf_mandelbrot_on_uniform_data():
    """Fitting uniform data should produce low R^2 or flat alpha."""
    tokens = list(range(100)) * 10  # uniform: each token appears 10 times
    ranks, freqs = compute_rank_frequency(tokens)
    result = fit_zipf_mandelbrot(ranks, freqs)
    # Uniform has near-zero variance in frequencies, so R^2 should be low
    # or alpha should be near 0
    assert result["r_squared"] < 0.5 or abs(result["alpha"]) < 0.3


def test_fit_zipf_mandelbrot_returns_required_keys():
    """Result dict must contain alpha, q, r_squared, C."""
    tokens = _generate_zipfian_tokens(alpha=1.0, n=5000, vocab_size=200)
    ranks, freqs = compute_rank_frequency(tokens)
    result = fit_zipf_mandelbrot(ranks, freqs)
    for key in ["alpha", "q", "r_squared", "C"]:
        assert key in result, f"Missing key: {key}"


def test_fit_piecewise_zipf_returns_regions():
    """Piecewise fit must return head, body, tail regions."""
    tokens = _generate_zipfian_tokens(alpha=1.0, n=10000, vocab_size=500)
    ranks, freqs = compute_rank_frequency(tokens)
    result = fit_piecewise_zipf(ranks, freqs)
    for region in ["head", "body", "tail"]:
        assert region in result, f"Missing region: {region}"
        assert "alpha" in result[region], f"Missing alpha for {region}"
        assert "r_squared" in result[region], f"Missing r_squared for {region}"


def test_detect_breakpoints_returns_list():
    """Breakpoint detection must return a list of rank indices."""
    tokens = _generate_zipfian_tokens(alpha=1.0, n=10000, vocab_size=500)
    ranks, freqs = compute_rank_frequency(tokens)
    breakpoints = detect_breakpoints(ranks, freqs)
    assert isinstance(breakpoints, list)
    # Breakpoints should be rank values (integers >= 1)
    for bp in breakpoints:
        assert isinstance(bp, (int, np.integer))
        assert bp >= 1


def test_analyze_corpus_returns_complete_result():
    """analyze_corpus must return dict with all analysis components."""
    tokens = _generate_zipfian_tokens(alpha=1.0, n=5000, vocab_size=200)
    result = analyze_corpus(tokens, "test_corpus")
    assert result["label"] == "test_corpus"
    assert "global_fit" in result
    assert "piecewise_fit" in result
    assert "breakpoints" in result
    assert "num_unique_tokens" in result
    assert "num_total_tokens" in result


def test_analyze_corpus_empty_tokens():
    """analyze_corpus on empty tokens should not crash."""
    result = analyze_corpus([], "empty")
    assert result["label"] == "empty"
    assert result["num_total_tokens"] == 0
