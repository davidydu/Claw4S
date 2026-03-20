# tests/test_metrics.py
import math
from collections import Counter
from src.metrics import compute_metrics, token_entropy

def test_token_entropy_uniform():
    """Uniform distribution over N tokens has entropy log2(N)."""
    tokens = [0, 1, 2, 3]
    assert abs(token_entropy(tokens) - 2.0) < 0.01

def test_token_entropy_single():
    """All same token has entropy 0."""
    tokens = [5, 5, 5, 5]
    assert token_entropy(tokens) == 0.0

def test_compute_metrics_basic():
    """Verify metric computation on known inputs."""
    result = compute_metrics(
        token_ids=[0, 1, 2, 0, 1, 2, 0, 1, 2, 0],  # 10 tokens, 3 unique
        num_characters=50,
        num_words=8,
        vocab_size=100000,
    )
    assert result["fertility"] == 10 / 8  # 1.25
    assert result["compression_ratio"] == 50 / 10  # 5.0
    assert result["vocab_utilization"] == 3 / 100000
    assert result["bpc"] > 0  # entropy-based, not zero

def test_compute_metrics_cross_lingual_tax():
    """Cross-lingual tax uses compression ratio, not fertility."""
    result = compute_metrics(
        token_ids=list(range(30)),
        num_characters=100,
        num_words=10,
        vocab_size=100000,
        baseline_compression=5.0,  # English gets 5 chars/token
    )
    # This lang gets 100/30 = 3.33 chars/token
    # Tax = baseline / lang = 5.0 / 3.33 = 1.5
    assert abs(result["cross_lingual_tax"] - 1.5) < 0.01
