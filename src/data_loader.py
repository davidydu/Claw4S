"""Download and load Tatoeba parallel sentence pairs."""

from datasets import load_dataset

DEFAULT_PAIRS = [
    "en-de", "en-fr", "en-es", "en-ru", "en-zh",
    "en-ja", "en-ko", "en-hi", "en-ar", "en-tr",
    "en-vi", "en-fi", "en-he",
]

# Map Tatoeba pair codes to readable language names
LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "hi": "Hindi", "ar": "Arabic", "tr": "Turkish", "vi": "Vietnamese",
    "fi": "Finnish", "he": "Hebrew",
}


def load_parallel_sentences(
    pairs: list[str] | None = None,
    max_sentences: int = 200,
) -> dict[str, str]:
    """Load Tatoeba parallel sentences, return {lang_code: concatenated_text}.

    For each pair (e.g., 'en-fr'), loads parallel English and target
    language sentences. English sentences are aggregated across all pairs
    to build a combined English baseline.
    """
    if pairs is None:
        pairs = DEFAULT_PAIRS

    samples: dict[str, list[str]] = {"en": []}

    for pair in pairs:
        target_lang = pair.split("-")[1]
        try:
            ds = load_dataset(
                "sentence-transformers/parallel-sentences-tatoeba",
                pair,
                split="train",
            )
            n = min(max_sentences, len(ds))
            subset = ds.select(range(n))

            en_sents = [row["english"] for row in subset]
            tgt_sents = [row["non_english"] for row in subset]

            samples["en"].extend(en_sents)
            samples[target_lang] = tgt_sents

            print(f"  Loaded {pair}: {n} sentence pairs")
        except Exception as e:
            print(f"  Warning: Could not load {pair}: {e}")

    # Concatenate sentence lists into text blocks
    return {lang: "\n".join(sents) for lang, sents in samples.items() if sents}
