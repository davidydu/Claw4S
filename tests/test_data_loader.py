from src.data_loader import load_parallel_sentences

def test_load_returns_dict_with_english_and_target():
    """Load a small sample and verify structure."""
    samples = load_parallel_sentences(["en-fr"], max_sentences=10)
    assert isinstance(samples, dict)
    assert "en" in samples  # English baseline
    assert "fr" in samples
    assert isinstance(samples["en"], str)
    assert isinstance(samples["fr"], str)
    assert len(samples["en"]) > 0
    assert len(samples["fr"]) > 0

def test_load_default_languages():
    """Should load all default language pairs."""
    samples = load_parallel_sentences(max_sentences=5)
    assert "en" in samples
    assert len(samples) >= 10  # at least most languages should load
