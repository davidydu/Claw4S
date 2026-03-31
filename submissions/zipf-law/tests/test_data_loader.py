# tests/test_data_loader.py
"""Tests for data loading functions."""

from src import data_loader


class FakeDataset:
    """Minimal dataset stub implementing len/select/iteration."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)

    def select(self, indices):
        return FakeDataset([self.rows[i] for i in indices])


def _fake_tatoeba_rows(n=10):
    return [
        {
            "english": f"English sentence {i}",
            "non_english": f"Target sentence {i}",
        }
        for i in range(n)
    ]


def _fake_code_rows(n=10):
    return [{"whole_func_string": f"def fn_{i}():\n    return {i}"} for i in range(n)]


def test_load_tatoeba_returns_expected_languages_and_caps_english(monkeypatch):
    """Tatoeba loader should return English + target language with capped English."""
    call_args = []

    def fake_load_dataset(name, pair, split, revision):
        call_args.append((name, pair, split, revision))
        return FakeDataset(_fake_tatoeba_rows(12))

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)

    result = data_loader.load_tatoeba_sentences(pairs=["en-de", "en-fr"], max_sentences=5)

    assert set(result.keys()) == {"en", "de", "fr"}
    assert len(result["en"].splitlines()) == 5
    assert len(result["de"].splitlines()) == 5
    assert len(result["fr"].splitlines()) == 5
    assert all(args[3] == data_loader.TATOEBA_REVISION for args in call_args)


def test_load_tatoeba_uses_pinned_revision(monkeypatch):
    """Tatoeba loader should pass the pinned dataset revision."""
    seen_revision = None

    def fake_load_dataset(name, pair, split, revision):
        nonlocal seen_revision
        seen_revision = revision
        return FakeDataset(_fake_tatoeba_rows(5))

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)
    data_loader.load_tatoeba_sentences(pairs=["en-ja"], max_sentences=3)

    assert seen_revision == data_loader.TATOEBA_REVISION


def test_load_code_samples_returns_nonempty_strings_and_uses_revision(monkeypatch):
    """Code loader should return joined code text and pinned revision."""
    seen_calls = []

    def fake_load_dataset(name, lang, split, revision):
        seen_calls.append((name, lang, split, revision))
        return FakeDataset(_fake_code_rows(8))

    monkeypatch.setattr(data_loader, "load_dataset", fake_load_dataset)

    result = data_loader.load_code_samples(languages=["python", "java"], max_samples=4)

    assert set(result.keys()) == {"python", "java"}
    assert "def fn_0()" in result["python"]
    assert "def fn_0()" in result["java"]
    assert all(call[3] == data_loader.CODESEARCHNET_REVISION for call in seen_calls)


def test_lang_names_and_code_languages_have_expected_entries():
    """Basic schema expectations for language metadata."""
    assert len(data_loader.LANG_NAMES) >= 10
    assert data_loader.LANG_NAMES["en"] == "English"
    assert len(data_loader.CODE_LANGUAGES) >= 2
    assert "python" in data_loader.CODE_LANGUAGES
