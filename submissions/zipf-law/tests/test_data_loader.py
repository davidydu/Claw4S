# tests/test_data_loader.py
"""Tests for data loading functions."""

from src.data_loader import (
    load_tatoeba_sentences,
    load_code_samples,
    LANG_NAMES,
    CODE_LANGUAGES,
)


def test_load_tatoeba_returns_dict_with_en():
    """Tatoeba loader must return a dict containing 'en' key."""
    result = load_tatoeba_sentences(pairs=["en-de"], max_sentences=5)
    assert isinstance(result, dict)
    assert "en" in result
    assert "de" in result


def test_load_tatoeba_returns_nonempty_strings():
    """Tatoeba sentences must be non-empty strings."""
    result = load_tatoeba_sentences(pairs=["en-fr"], max_sentences=5)
    for lang, text in result.items():
        assert isinstance(text, str)
        assert len(text) > 0, f"Empty text for language {lang}"


def test_load_tatoeba_caps_english():
    """English text should be capped to max_sentences, not accumulated."""
    result = load_tatoeba_sentences(
        pairs=["en-de", "en-fr"], max_sentences=5
    )
    en_lines = result["en"].strip().split("\n")
    assert len(en_lines) <= 5


def test_load_code_samples_returns_dict():
    """Code loader must return dict with language keys."""
    result = load_code_samples(languages=["python"], max_samples=5)
    assert isinstance(result, dict)
    assert "python" in result


def test_load_code_samples_nonempty():
    """Code samples must be non-empty strings."""
    result = load_code_samples(languages=["python"], max_samples=5)
    for lang, text in result.items():
        assert isinstance(text, str)
        assert len(text) > 0, f"Empty code for language {lang}"


def test_lang_names_has_entries():
    """LANG_NAMES should map codes to readable names."""
    assert len(LANG_NAMES) >= 10
    assert LANG_NAMES["en"] == "English"


def test_code_languages_has_entries():
    """CODE_LANGUAGES should list available code languages."""
    assert len(CODE_LANGUAGES) >= 2
    assert "python" in CODE_LANGUAGES
