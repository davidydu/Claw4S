# src/data_loader.py
"""Download and load text corpora for Zipf analysis.

Supports two corpus types:
- Natural language: Tatoeba parallel sentences (same as tokenizer-analysis)
- Code: Python and Java code from CodeSearchNet
"""

from datasets import load_dataset

# Tatoeba language pairs (reuse from tokenizer-analysis)
DEFAULT_PAIRS = [
    "en-de", "en-fr", "en-es", "en-ru", "en-zh",
    "en-ja", "en-ko", "en-hi", "en-ar", "en-tr",
    "en-vi", "en-fi", "en-he",
]

LANG_NAMES = {
    "en": "English", "de": "German", "fr": "French", "es": "Spanish",
    "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "hi": "Hindi", "ar": "Arabic", "tr": "Turkish", "vi": "Vietnamese",
    "fi": "Finnish", "he": "Hebrew",
}

# Code languages available in CodeSearchNet
CODE_LANGUAGES = ["python", "java"]

# Pinned dataset revisions for reproducibility
TATOEBA_REVISION = "cec1343ab5a7a8befe99af4a2d0ca847b6c84743"
CODESEARCHNET_REVISION = "bd0cf261e357a3eb5c8fba490d23ec1a1cd59555"


def load_tatoeba_sentences(
    pairs: list[str] | None = None,
    max_sentences: int = 200,
) -> dict[str, str]:
    """Load Tatoeba parallel sentences, return {lang_code: concatenated_text}.

    For each pair (e.g., 'en-fr'), loads parallel English and target
    language sentences. English sentences are capped to max_sentences
    to match target language corpus sizes.
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
                revision=TATOEBA_REVISION,
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

    # Cap English to max_sentences to match target corpus sizes
    samples["en"] = samples["en"][:max_sentences]

    return {lang: "\n".join(sents) for lang, sents in samples.items() if sents}


def load_code_samples(
    languages: list[str] | None = None,
    max_samples: int = 200,
) -> dict[str, str]:
    """Load code samples from CodeSearchNet, return {language: concatenated_code}.

    Uses the 'whole_func_string' field which contains complete function bodies.
    """
    if languages is None:
        languages = CODE_LANGUAGES

    samples: dict[str, str] = {}

    for lang in languages:
        try:
            ds = load_dataset(
                "code_search_net",
                lang,
                split="train",
                revision=CODESEARCHNET_REVISION,
            )
            n = min(max_samples, len(ds))
            subset = ds.select(range(n))

            code_snippets = [row["whole_func_string"] for row in subset]
            samples[lang] = "\n\n".join(code_snippets)

            print(f"  Loaded {lang}: {n} code samples")
        except Exception as e:
            print(f"  Warning: Could not load {lang}: {e}")

    return samples
