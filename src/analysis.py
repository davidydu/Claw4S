# src/analysis.py
"""Run the full cross-lingual tokenizer analysis pipeline."""

import json
import os
from datetime import datetime, timezone

from src.data_loader import load_parallel_sentences, LANG_NAMES
from src.metrics import compute_metrics
from src.tokenizer_manager import load_tokenizers, tokenize


def run_analysis(
    pairs: list[str] | None = None,
    max_sentences: int = 200,
    output_dir: str = "results",
) -> dict:
    """Run full analysis: load data, tokenize, compute metrics, save results."""

    print("=" * 60)
    print("Cross-Lingual Tokenizer Analysis")
    print("=" * 60)

    # Step 1: Load corpus
    print("\n[1/4] Loading Tatoeba parallel sentences...")
    samples = load_parallel_sentences(pairs, max_sentences)
    print(f"  Loaded {len(samples)} languages")

    # Step 2: Load tokenizers
    print("\n[2/4] Loading tokenizers...")
    tokenizers = load_tokenizers()
    print(f"  Loaded {len(tokenizers)} tokenizers")

    if not tokenizers:
        raise RuntimeError("No tokenizers could be loaded")

    # Step 3: Compute metrics
    print("\n[3/4] Computing metrics...")
    all_results = []
    english_compressions = {}  # tokenizer_name -> english compression ratio

    # First pass: compute English compression baselines
    if "en" in samples:
        en_text = samples["en"]
        en_chars = len(en_text)
        for tok_name, tok_entry in tokenizers.items():
            tokens = tokenize(tok_entry, en_text)
            english_compressions[tok_name] = en_chars / len(tokens) if tokens else 1.0

    # Second pass: compute all metrics
    for tok_name, tok_entry in tokenizers.items():
        baseline_comp = english_compressions.get(tok_name)
        for lang, text in samples.items():
            tokens = tokenize(tok_entry, text)
            num_chars = len(text)
            num_words = len(text.split())

            m = compute_metrics(
                token_ids=tokens,
                num_characters=num_chars,
                num_words=num_words,
                vocab_size=tok_entry["vocab_size"],
                baseline_compression=baseline_comp,
            )

            lang_name = LANG_NAMES.get(lang, lang)
            all_results.append({
                "tokenizer": tok_name,
                "language": lang,
                "language_name": lang_name,
                **m,
            })
            print(f"  {tok_name} x {lang_name}: "
                  f"compression={m['compression_ratio']:.2f}, "
                  f"tax={m['cross_lingual_tax']:.2f}x")

    # Step 4: Save results
    print(f"\n[4/4] Saving results to {output_dir}/")
    os.makedirs(output_dir, exist_ok=True)

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_languages": len(samples),
            "num_tokenizers": len(tokenizers),
            "max_sentences": max_sentences,
            "tokenizers": {
                name: {"vocab_size": entry["vocab_size"]}
                for name, entry in tokenizers.items()
            },
            "languages": list(samples.keys()),
        },
        "results": all_results,
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved {results_path}")

    return output
