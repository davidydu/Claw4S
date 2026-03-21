"""Run the full Zipf's law analysis pipeline.

Orchestrates data loading, tokenization, Zipf fitting, plotting, and
report generation. All results are saved to results/.

Must be run from the submissions/zipf-law/ directory.
"""

import json
import os
import sys
import traceback
from datetime import datetime, timezone

import numpy as np
from scipy import stats

# Working directory guard
expected_marker = os.path.join("src", "zipf_analysis.py")
if not os.path.exists(expected_marker):
    print(f"ERROR: Must run from submissions/zipf-law/ directory.")
    print(f"  Expected to find: {expected_marker}")
    sys.exit(1)

os.makedirs("results/figures", exist_ok=True)

from src.data_loader import load_tatoeba_sentences, load_code_samples, LANG_NAMES
from src.tokenizer_manager import load_tokenizers, tokenize
from src.zipf_analysis import compute_rank_frequency, analyze_corpus
from src.plots import (
    plot_zipf_fit,
    plot_piecewise_comparison,
    plot_alpha_compression_correlation,
    plot_zipf_overlay,
)
from src.report import generate_report

SEED = 42
np.random.seed(SEED)


def main():
    print("=" * 60)
    print("Zipf's Law Breakdown in Token Distributions")
    print("=" * 60)

    # --- Step 1: Load corpora ---
    print("\n[1/6] Loading corpora...")

    # Natural language (selected languages for manageable runtime)
    nl_pairs = ["en-de", "en-fr", "en-zh", "en-ja", "en-ar", "en-fi"]
    nl_samples = load_tatoeba_sentences(pairs=nl_pairs, max_sentences=200)
    print(f"  Natural language: {len(nl_samples)} languages")

    # Code
    code_samples = load_code_samples(languages=["python", "java"], max_samples=200)
    print(f"  Code: {len(code_samples)} languages")

    if not nl_samples:
        print("ERROR: No natural language data loaded.")
        sys.exit(1)

    # --- Step 2: Load tokenizers ---
    print("\n[2/6] Loading tokenizers...")
    tokenizers = load_tokenizers()
    print(f"  Loaded {len(tokenizers)} tokenizers")

    if len(tokenizers) < 2:
        print("ERROR: Need at least 2 tokenizers.")
        sys.exit(1)

    # --- Step 3: Tokenize and analyze ---
    print("\n[3/6] Tokenizing and analyzing Zipf distributions...")

    all_analyses = []
    overlay_data = []  # For overlay plot

    # Natural language corpora
    for lang, text in nl_samples.items():
        lang_name = LANG_NAMES.get(lang, lang)
        for tok_name, tok_entry in tokenizers.items():
            try:
                tokens = tokenize(tok_entry, text)
                if not tokens:
                    print(f"  WARNING: No tokens for {lang_name} x {tok_name}")
                    continue

                label = f"{lang_name} ({tok_name})"
                result = analyze_corpus(tokens, label)

                # Compute compression ratio
                compression = len(text) / len(tokens) if tokens else 0.0

                result["tokenizer"] = tok_name
                result["corpus"] = lang_name
                result["corpus_type"] = "natural_language"
                result["language"] = lang
                result["compression_ratio"] = compression

                all_analyses.append(result)

                gf = result["global_fit"]
                print(
                    f"  {label}: alpha={gf['alpha']:.3f}, "
                    f"R^2={gf['r_squared']:.4f}, "
                    f"compression={compression:.2f}"
                )

                # Collect for overlay (limit to English for clarity)
                if lang == "en":
                    ranks, freqs = compute_rank_frequency(tokens)
                    overlay_data.append({
                        "label": f"English ({tok_name})",
                        "ranks": ranks,
                        "freqs": freqs,
                    })

            except Exception as e:
                print(f"  ERROR: {lang_name} x {tok_name}: {e}")
                traceback.print_exc()

    # Code corpora
    for code_lang, code_text in code_samples.items():
        for tok_name, tok_entry in tokenizers.items():
            try:
                tokens = tokenize(tok_entry, code_text)
                if not tokens:
                    print(f"  WARNING: No tokens for {code_lang} x {tok_name}")
                    continue

                label = f"{code_lang.capitalize()} code ({tok_name})"
                result = analyze_corpus(tokens, label)

                compression = len(code_text) / len(tokens) if tokens else 0.0

                result["tokenizer"] = tok_name
                result["corpus"] = f"{code_lang.capitalize()} code"
                result["corpus_type"] = "code"
                result["language"] = code_lang
                result["compression_ratio"] = compression

                all_analyses.append(result)

                gf = result["global_fit"]
                print(
                    f"  {label}: alpha={gf['alpha']:.3f}, "
                    f"R^2={gf['r_squared']:.4f}, "
                    f"compression={compression:.2f}"
                )

                # Add code to overlay
                ranks, freqs = compute_rank_frequency(tokens)
                overlay_data.append({
                    "label": f"{code_lang.capitalize()} ({tok_name})",
                    "ranks": ranks,
                    "freqs": freqs,
                })

            except Exception as e:
                print(f"  ERROR: {code_lang} x {tok_name}: {e}")
                traceback.print_exc()

    if not all_analyses:
        print("ERROR: No analyses completed.")
        sys.exit(1)

    # --- Step 4: Correlation analysis ---
    print("\n[4/6] Computing correlation between Zipf exponent and compression...")

    alphas = [a["global_fit"]["alpha"] for a in all_analyses]
    compressions = [a["compression_ratio"] for a in all_analyses]

    correlation = {}
    if len(alphas) >= 3:
        pearson_r, pearson_p = stats.pearsonr(alphas, compressions)
        spearman_r, spearman_p = stats.spearmanr(alphas, compressions)
        correlation = {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_r": float(spearman_r),
            "spearman_p": float(spearman_p),
        }
        print(
            f"  Pearson r = {pearson_r:.4f} (p = {pearson_p:.4f})"
        )
        print(
            f"  Spearman rho = {spearman_r:.4f} (p = {spearman_p:.4f})"
        )

    # --- Step 5: Generate plots ---
    print("\n[5/6] Generating figures...")

    # Individual Zipf fit plots (select a few representative cases)
    plot_cases = []
    seen = set()
    for a in all_analyses:
        key = a["corpus_type"]
        if key not in seen and len(plot_cases) < 4:
            seen.add(key)
            plot_cases.append(a)

    for a in plot_cases:
        # Re-tokenize to get ranks/freqs for plotting
        safe_name = a["label"].replace(" ", "_").replace("(", "").replace(")", "")
        ranks_data = None
        freqs_data = None

        # Find matching overlay data
        for od in overlay_data:
            if od["label"] == a["label"]:
                ranks_data = od["ranks"]
                freqs_data = od["freqs"]
                break

        if ranks_data is not None:
            plot_zipf_fit(
                ranks_data,
                freqs_data,
                a["global_fit"],
                a["label"],
                f"results/figures/zipf_fit_{safe_name}.png",
            )

    # Piecewise comparison
    plot_piecewise_comparison(
        all_analyses,
        "results/figures/piecewise_exponents.png",
    )

    # Correlation scatter
    labels = [a["label"] for a in all_analyses]
    plot_alpha_compression_correlation(
        alphas,
        compressions,
        labels,
        "results/figures/correlation_alpha_compression.png",
    )

    # Overlay plot
    if overlay_data:
        plot_zipf_overlay(
            overlay_data,
            "results/figures/zipf_overlay.png",
        )

    print(f"  Generated figures in results/figures/")

    # --- Step 6: Save results and generate report ---
    print("\n[6/6] Saving results and generating report...")

    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    # Clean analyses for JSON serialization (remove numpy arrays)
    clean_analyses = []
    for a in all_analyses:
        clean = {}
        for k, v in a.items():
            if isinstance(v, dict):
                clean[k] = {
                    kk: convert_numpy(vv) if not isinstance(vv, dict)
                    else {kkk: convert_numpy(vvv) for kkk, vvv in vv.items()}
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                clean[k] = [convert_numpy(x) for x in v]
            else:
                clean[k] = convert_numpy(v)
        clean_analyses.append(clean)

    output = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_tokenizers": len(tokenizers),
            "num_corpora": len(nl_samples) + len(code_samples),
            "seed": SEED,
            "tokenizers": {
                name: {"vocab_size": entry["vocab_size"]}
                for name, entry in tokenizers.items()
            },
            "natural_languages": list(nl_samples.keys()),
            "code_languages": list(code_samples.keys()),
        },
        "analyses": clean_analyses,
        "correlation": correlation,
    }

    with open("results/results.json", "w") as f:
        json.dump(output, f, indent=2, default=convert_numpy)
    print("  Saved results/results.json")

    report = generate_report(output)
    print("  Saved results/report.md")

    print("\n" + "=" * 60)
    print("Analysis complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
