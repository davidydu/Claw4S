"""Information-theoretic metrics for tokenizer analysis."""

import math
from collections import Counter


def token_entropy(token_ids: list[int]) -> float:
    """Compute Shannon entropy of the token distribution in bits.

    H = -sum(p_i * log2(p_i)) where p_i is the empirical frequency
    of each token in the sequence.
    """
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    total = len(token_ids)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_metrics(
    token_ids: list[int],
    num_characters: int,
    num_words: int,
    vocab_size: int,
    baseline_compression: float | None = None,
) -> dict:
    """Compute all metrics for a single (tokenizer, language) pair."""
    num_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))

    fertility = num_tokens / num_words if num_words > 0 else 0.0
    compression_ratio = num_characters / num_tokens if num_tokens > 0 else 0.0
    vocab_utilization = unique_tokens / vocab_size if vocab_size > 0 else 0.0

    entropy = token_entropy(token_ids)
    bpc = (entropy * num_tokens) / num_characters if num_characters > 0 else 0.0

    if baseline_compression and baseline_compression > 0 and compression_ratio > 0:
        cross_lingual_tax = baseline_compression / compression_ratio
    else:
        cross_lingual_tax = 1.0

    return {
        "fertility": fertility,
        "bpc": bpc,
        "compression_ratio": compression_ratio,
        "cross_lingual_tax": cross_lingual_tax,
        "vocab_utilization": vocab_utilization,
        "token_entropy": entropy,
    }
