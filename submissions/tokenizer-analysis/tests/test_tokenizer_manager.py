# tests/test_tokenizer_manager.py
from src.tokenizer_manager import load_tokenizers, tokenize

def test_load_tokenizers_returns_at_least_two():
    """Should always load tiktoken tokenizers (no auth needed)."""
    tokenizers = load_tokenizers()
    assert len(tokenizers) >= 2
    assert "gpt4o" in tokenizers
    assert "gpt4" in tokenizers

def test_tokenize_returns_token_ids():
    """Tokenize a simple string and get back integer token IDs."""
    tokenizers = load_tokenizers()
    tokens = tokenize(tokenizers["gpt4o"], "Hello world")
    assert isinstance(tokens, list)
    assert all(isinstance(t, int) for t in tokens)
    assert len(tokens) > 0
