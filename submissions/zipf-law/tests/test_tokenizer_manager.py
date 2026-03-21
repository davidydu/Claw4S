# tests/test_tokenizer_manager.py
"""Tests for tokenizer loading and tokenization."""

from src.tokenizer_manager import load_tokenizers, tokenize, TOKENIZER_CONFIGS


def test_load_tokenizers_returns_at_least_two():
    """Must load at least 2 tokenizers (tiktoken ones always available)."""
    loaded = load_tokenizers()
    assert len(loaded) >= 2, f"Expected >= 2 tokenizers, got {len(loaded)}"


def test_tokenize_returns_list_of_ints():
    """tokenize() must return a list of integers."""
    loaded = load_tokenizers()
    tok_name = next(iter(loaded))
    tokens = tokenize(loaded[tok_name], "Hello, world!")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0


def test_each_tokenizer_has_vocab_size():
    """Each loaded tokenizer must have vocab_size > 0."""
    loaded = load_tokenizers()
    for name, entry in loaded.items():
        assert "vocab_size" in entry
        assert entry["vocab_size"] > 0, f"{name} has vocab_size <= 0"


def test_tokenizer_configs_has_entries():
    """TOKENIZER_CONFIGS should define at least 2 tokenizers."""
    assert len(TOKENIZER_CONFIGS) >= 2


def test_tokenize_empty_string():
    """Tokenizing empty string should return empty list."""
    loaded = load_tokenizers()
    tok_name = next(iter(loaded))
    tokens = tokenize(loaded[tok_name], "")
    assert isinstance(tokens, list)
    assert len(tokens) == 0


def test_tokenize_multilingual():
    """Tokenizers must handle non-ASCII text without errors."""
    loaded = load_tokenizers()
    tok_name = next(iter(loaded))
    tokens = tokenize(loaded[tok_name], "Bonjour le monde! Hallo Welt!")
    assert len(tokens) > 0
