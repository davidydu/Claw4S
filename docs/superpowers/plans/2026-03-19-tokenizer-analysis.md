# Cross-Lingual Tokenizer Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an executable SKILL.md that performs cross-lingual tokenizer analysis comparing modern LLM tokenizers across 14 languages using information-theoretic metrics, then submit to Claw4S via clawRxiv.

**Architecture:** A modular Python pipeline orchestrated by SKILL.md. Three core modules — data loading (Tatoeba parallel sentences from HuggingFace, ungated), tokenizer management (tiktoken + HuggingFace transformers), and metrics computation (fertility, token entropy, compression ratio, vocabulary utilization). The SKILL.md walks the agent through: venv setup → install deps → download corpus → run analysis → generate report. All commands use `.venv/bin/python` directly (no `source activate` — shell state doesn't persist between agent tool calls). The research note summarizes findings in 1–4 page LaTeX.

**Tech Stack:** Python 3.10+, tiktoken, transformers, sentencepiece, datasets (HuggingFace), numpy, scipy

---

## File Structure

```
Claw4S/
├── SKILL.md                          # The executable skill (primary deliverable)
├── requirements.txt                  # Pinned dependencies
├── conftest.py                       # Ensures pytest finds src/ package
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Download Tatoeba parallel sentences
│   ├── tokenizer_manager.py          # Load tokenizers, handle auth gracefully
│   ├── metrics.py                    # Compute information-theoretic metrics
│   ├── analysis.py                   # Run full analysis pipeline
│   └── report.py                     # Generate markdown results report
├── tests/
│   ├── test_metrics.py               # Unit tests for metric calculations
│   ├── test_data_loader.py           # Test corpus loading
│   └── test_analysis.py              # Integration test for pipeline
├── research_note/
│   └── main.tex                      # 1–4 page LaTeX research note
├── .venv/                            # Created at runtime (gitignored)
├── results/                          # Created at runtime (gitignored)
│   ├── results.json                  # Raw metrics data
│   └── report.md                     # Human/agent-readable summary
└── docs/superpowers/plans/           # This plan
```

## Data Source

**Dataset:** `sentence-transformers/parallel-sentences-tatoeba` (ungated, no auth required)

This provides parallel English + target language sentence pairs. Each config (e.g., `en-fr`) gives rows with `{"english": "...", "non_english": "..."}`. We load 200 sentence pairs per language and tokenize both the English and target text.

**Why not FLORES+?** `openlanguagedata/flores_plus` is gated (requires HuggingFace auth). The review agent won't have a token. Tatoeba is ungated and provides true parallel text for cross-lingual comparison.

## Tokenizers

Use only tokenizers that load **without HuggingFace authentication**:

| ID | Tokenizer | Package | Vocab Size | Auth Required |
|----|-----------|---------|-----------|---------------|
| `gpt4o` | o200k_base | tiktoken | ~200K | No |
| `gpt4` | cl100k_base | tiktoken | ~100K | No |
| `mistral` | mistralai/Mistral-7B-v0.1 | transformers | ~32K | No |
| `qwen2.5` | Qwen/Qwen2.5-7B | transformers | ~152K | No |
| `gemma2` | google/gemma-2-2b | transformers | ~256K | Possibly — load with fallback |
| `llama3` | meta-llama/Meta-Llama-3-8B | transformers | ~128K | Possibly — load with fallback |

The pipeline gracefully skips any tokenizer that fails to load.

## Languages

14 languages from Tatoeba spanning major language families (verified accessible):

| Tatoeba Config | Language | Family | Script |
|------|----------|--------|--------|
| (baseline) | English | Germanic | Latin |
| en-de | German | Germanic | Latin |
| en-fr | French | Romance | Latin |
| en-es | Spanish | Romance | Latin |
| en-ru | Russian | Slavic | Cyrillic |
| en-zh | Chinese | Sinitic | CJK |
| en-ja | Japanese | Japonic | CJK+Kana |
| en-ko | Korean | Koreanic | Hangul |
| en-hi | Hindi | Indo-Aryan | Devanagari |
| en-ar | Arabic | Semitic | Arabic |
| en-tr | Turkish | Turkic | Latin |
| en-vi | Vietnamese | Austroasiatic | Latin |
| en-fi | Finnish | Uralic | Latin |
| en-he | Hebrew | Semitic | Hebrew |

## Metrics

For each (tokenizer, language) pair, compute:

1. **Fertility** — `num_tokens / num_words` (tokens per word; note: meaningful only for space-delimited languages — CJK values are reported but flagged)
2. **Token entropy (BPC)** — `-sum(p_i × log2(p_i)) × num_tokens / num_characters` where p_i is empirical token frequency (actual bits-per-character, not uniform upper bound)
3. **Compression ratio** — `num_characters / num_tokens` (chars per token — higher is better; primary cross-lingual metric since it's script-agnostic)
4. **Cross-lingual tax** — `compression_ratio(English) / compression_ratio(lang)` (>1.0 means the language is "taxed"; uses compression ratio instead of fertility to handle CJK correctly)
5. **Vocabulary utilization** — `unique_tokens_used / vocab_size` (what fraction of vocab is activated)

---

## Task 1: Project Setup

**Files:**
- Modify: `.gitignore`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `conftest.py`

- [ ] **Step 1: Update .gitignore**

Add to existing `.gitignore`:
```
.venv/
results/
__pycache__/
*.pyc
*.egg-info/
```

- [ ] **Step 2: Create requirements.txt**

```
tiktoken>=0.7.0
transformers>=4.40.0
sentencepiece>=0.1.99
protobuf>=3.20.0
datasets>=2.19.0
numpy>=1.24.0
scipy>=1.11.0
pytest>=7.0.0
```

- [ ] **Step 3: Create src/__init__.py**

Empty file.

- [ ] **Step 4: Create conftest.py**

```python
# conftest.py — ensures pytest can import from src/
```

Empty file at project root. Presence makes pytest treat the root as rootdir.

- [ ] **Step 5: Create venv and install**

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

- [ ] **Step 6: Verify installs**

```bash
.venv/bin/python -c "import tiktoken, transformers, datasets, numpy, scipy, sentencepiece; print('All imports OK')"
```

Expected: `All imports OK`

- [ ] **Step 7: Commit**

```bash
git add .gitignore requirements.txt src/__init__.py conftest.py
git commit -m "feat: project setup with dependencies and gitignore"
```

---

## Task 2: Data Loader Module

**Files:**
- Create: `src/data_loader.py`
- Create: `tests/test_data_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data_loader.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_data_loader.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.data_loader'`

- [ ] **Step 3: Implement data_loader.py**

```python
# src/data_loader.py
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_data_loader.py -v
```

Expected: 2 passed (may take a moment to download on first run)

- [ ] **Step 5: Commit**

```bash
git add src/data_loader.py tests/test_data_loader.py
git commit -m "feat: Tatoeba parallel sentence data loader"
```

---

## Task 3: Tokenizer Manager Module

**Files:**
- Create: `src/tokenizer_manager.py`
- Create: `tests/test_tokenizer_manager.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_tokenizer_manager.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement tokenizer_manager.py**

```python
# src/tokenizer_manager.py
"""Load multiple tokenizers with graceful fallback for gated models."""

import tiktoken

TOKENIZER_CONFIGS = {
    "gpt4o": {"type": "tiktoken", "encoding": "o200k_base"},
    "gpt4": {"type": "tiktoken", "encoding": "cl100k_base"},
    "mistral": {"type": "hf", "model": "mistralai/Mistral-7B-v0.1"},
    "qwen2.5": {"type": "hf", "model": "Qwen/Qwen2.5-7B"},
    "gemma2": {"type": "hf", "model": "google/gemma-2-2b"},
    "llama3": {"type": "hf", "model": "meta-llama/Meta-Llama-3-8B"},
}


def load_tokenizers() -> dict:
    """Load all available tokenizers. Skip gated ones that require auth."""
    loaded = {}
    for name, config in TOKENIZER_CONFIGS.items():
        try:
            if config["type"] == "tiktoken":
                tok = tiktoken.get_encoding(config["encoding"])
                loaded[name] = {"tokenizer": tok, "type": "tiktoken",
                                "vocab_size": tok.n_vocab}
            else:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(
                    config["model"], trust_remote_code=True
                )
                loaded[name] = {"tokenizer": tok, "type": "hf",
                                "vocab_size": tok.vocab_size}
            print(f"  Loaded {name} (vocab: {loaded[name]['vocab_size']:,})")
        except Exception as e:
            print(f"  Skipping {name}: {e}")
    return loaded


def tokenize(tok_entry: dict, text: str) -> list[int]:
    """Tokenize text, returning list of token IDs."""
    tok = tok_entry["tokenizer"]
    if tok_entry["type"] == "tiktoken":
        return tok.encode(text)
    else:
        return tok.encode(text, add_special_tokens=False)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_tokenizer_manager.py -v
```

Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/tokenizer_manager.py tests/test_tokenizer_manager.py
git commit -m "feat: tokenizer manager with graceful fallback for gated models"
```

---

## Task 4: Metrics Module

**Files:**
- Create: `src/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_metrics.py
import math
from collections import Counter
from src.metrics import compute_metrics, token_entropy

def test_token_entropy_uniform():
    """Uniform distribution over N tokens has entropy log2(N)."""
    # 4 distinct tokens, each appears once → entropy = log2(4) = 2.0
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_metrics.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement metrics.py**

```python
# src/metrics.py
"""Information-theoretic metrics for tokenizer analysis."""

import math
from collections import Counter


def token_entropy(token_ids: list[int]) -> float:
    """Compute Shannon entropy of the token distribution in bits.

    H = -sum(p_i * log2(p_i)) where p_i is the empirical frequency
    of each token in the sequence.
    """
    if not token_ids:
        return 0.0
    counts = Counter(token_ids)
    total = len(token_ids)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def compute_metrics(
    token_ids: list[int],
    num_characters: int,
    num_words: int,
    vocab_size: int,
    baseline_compression: float | None = None,
) -> dict:
    """Compute all metrics for a single (tokenizer, language) pair.

    Args:
        token_ids: Token IDs produced by the tokenizer.
        num_characters: Total characters in the input text.
        num_words: Total whitespace-delimited words in the input text.
        vocab_size: Tokenizer vocabulary size.
        baseline_compression: English compression ratio for cross-lingual tax.
            If None, cross_lingual_tax is set to 1.0.

    Returns:
        Dict with keys: fertility, bpc, compression_ratio,
        cross_lingual_tax, vocab_utilization.
    """
    num_tokens = len(token_ids)
    unique_tokens = len(set(token_ids))

    fertility = num_tokens / num_words if num_words > 0 else 0.0
    compression_ratio = num_characters / num_tokens if num_tokens > 0 else 0.0
    vocab_utilization = unique_tokens / vocab_size if vocab_size > 0 else 0.0

    # BPC: actual entropy of token distribution, scaled to bits per character
    entropy = token_entropy(token_ids)
    bpc = (entropy * num_tokens) / num_characters if num_characters > 0 else 0.0

    # Cross-lingual tax based on compression ratio (works for all scripts)
    if baseline_compression and baseline_compression > 0 and compression_ratio > 0:
        cross_lingual_tax = baseline_compression / compression_ratio
    else:
        cross_lingual_tax = 1.0

    return {
        "fertility": fertility,
        "bpc": bpc,
        "compression_ratio": compression_ratio,
        "cross_lingual_tax": cross_lingual_tax,
        "vocab_utilization": vocab_utilization,
        "token_entropy": entropy,
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_metrics.py -v
```

Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/metrics.py tests/test_metrics.py
git commit -m "feat: information-theoretic metrics with proper entropy-based BPC"
```

---

## Task 5: Analysis Pipeline

**Files:**
- Create: `src/analysis.py`
- Create: `tests/test_analysis.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis.py
import os
from src.analysis import run_analysis

def test_run_analysis_returns_structured_results():
    """Integration test: run analysis on minimal data."""
    results = run_analysis(
        pairs=["en-fr"],
        max_sentences=10,
        output_dir="/tmp/tokenizer_test_results",
    )
    assert "metadata" in results
    assert "results" in results
    assert len(results["results"]) > 0
    first = results["results"][0]
    assert "tokenizer" in first
    assert "language" in first
    assert "fertility" in first
    assert "bpc" in first
    assert "cross_lingual_tax" in first
    # Verify output file was written
    assert os.path.exists("/tmp/tokenizer_test_results/results.json")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_analysis.py -v
```

Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement analysis.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/python -m pytest tests/test_analysis.py -v
```

Expected: 1 passed (may take 30-60s for first data download)

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: analysis pipeline wiring data loader, tokenizers, and metrics"
```

---

## Task 6: Report Generator

**Files:**
- Create: `src/report.py`

- [ ] **Step 1: Implement report.py**

```python
# src/report.py
"""Generate a markdown report from analysis results."""


def generate_report(results: dict, output_path: str = "results/report.md") -> str:
    """Generate a markdown summary report from analysis results."""
    meta = results["metadata"]
    data = results["results"]

    lines = [
        "# Cross-Lingual Tokenizer Analysis Report",
        "",
        f"**Generated:** {meta['timestamp']}",
        f"**Tokenizers:** {meta['num_tokenizers']}",
        f"**Languages:** {meta['num_languages']}",
        f"**Sentences per language:** {meta['max_sentences']}",
        "",
    ]

    tokenizer_names = sorted(set(r["tokenizer"] for r in data))
    language_codes = meta["languages"]

    header = "| Language | " + " | ".join(tokenizer_names) + " |"
    sep = "|---|" + "|".join(["---"] * len(tokenizer_names)) + "|"

    def make_table(title, key, fmt=".2f", suffix=""):
        lines.append(f"## {title}")
        lines.append("")
        lines.extend([header, sep])
        for lang in language_codes:
            name = next((r["language_name"] for r in data
                         if r["language"] == lang), lang)
            row = f"| {name} ({lang}) |"
            for tok in tokenizer_names:
                match = [r for r in data
                         if r["tokenizer"] == tok and r["language"] == lang]
                if match:
                    row += f" {match[0][key]:{fmt}}{suffix} |"
                else:
                    row += " — |"
            lines.append(row)
        lines.append("")

    make_table("Compression Ratio (characters per token)", "compression_ratio")
    make_table("Cross-Lingual Tax (>1.0 = taxed vs English)", "cross_lingual_tax",
               fmt=".2f", suffix="x")
    make_table("Token Entropy (bits)", "token_entropy")
    make_table("Fertility (tokens per word)", "fertility",
               fmt=".2f")

    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append("### Tokenizer Equity Ranking (lower avg tax = more equitable)")
    lines.append("")

    rankings = []
    for tok in tokenizer_names:
        non_en = [r for r in data
                  if r["tokenizer"] == tok and r["language"] != "en"]
        if non_en:
            taxes = [r["cross_lingual_tax"] for r in non_en]
            avg_tax = sum(taxes) / len(taxes)
            worst = max(non_en, key=lambda r: r["cross_lingual_tax"])
            rankings.append((tok, avg_tax, worst))

    rankings.sort(key=lambda x: x[1])
    for tok, avg_tax, worst in rankings:
        lines.append(
            f"- **{tok}**: avg tax = {avg_tax:.2f}x, "
            f"max tax = {worst['cross_lingual_tax']:.2f}x "
            f"({worst['language_name']})"
        )

    lines.append("")
    lines.append("### Notes")
    lines.append("")
    lines.append("- Fertility (tokens/word) is unreliable for CJK languages "
                 "(Chinese, Japanese, Korean) because they don't use spaces. "
                 "Use compression ratio as the primary cross-lingual metric.")
    lines.append("- Cross-lingual tax uses compression ratio: "
                 "`tax = English_compression / language_compression`. "
                 "A tax of 2.0x means the language uses 2x more tokens per "
                 "character than English.")
    lines.append("")

    report = "\n".join(lines)

    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"Report saved to {output_path}")

    return report
```

- [ ] **Step 2: Quick smoke test**

```bash
.venv/bin/python -c "
from src.analysis import run_analysis
from src.report import generate_report
results = run_analysis(pairs=['en-fr', 'en-zh'], max_sentences=20,
                       output_dir='/tmp/tokenizer_smoke')
report = generate_report(results, '/tmp/tokenizer_smoke/report.md')
print(report[:600])
"
```

Expected: Markdown tables printed to stdout

- [ ] **Step 3: Commit**

```bash
git add src/report.py
git commit -m "feat: markdown report generator with equity ranking"
```

---

## Task 7: Write the SKILL.md

**Files:**
- Create: `SKILL.md`

This is the **primary deliverable** for Claw4S. It must be self-contained instructions an agent follows. All commands use `.venv/bin/python` directly — no `source activate` because shell state doesn't persist between agent tool calls.

- [ ] **Step 1: Write SKILL.md**

The SKILL.md should contain (write with actual backticks, not escaped):

```
---
name: cross-lingual-tokenizer-analysis
description: Analyze cross-lingual tokenizer efficiency across modern LLMs.
  Compares compression ratios, fertility rates, entropy, and cross-lingual
  tax for GPT-4o, Mistral, Qwen, and other tokenizers across 14 languages
  using Tatoeba parallel sentences.
allowed-tools: Bash(python *), Bash(pip *), Bash(.venv/*), Read, Write
---
```

**Step 1: Environment Setup**
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```
Verify:
```bash
.venv/bin/python -c "import tiktoken, transformers, datasets, numpy, scipy, sentencepiece; print('All imports OK')"
```

**Step 2: Run Unit Tests**
```bash
.venv/bin/python -m pytest tests/ -v
```

**Step 3: Run the Analysis**
```bash
.venv/bin/python -c "
from src.analysis import run_analysis
from src.report import generate_report
results = run_analysis(max_sentences=200)
report = generate_report(results)
print(report)
"
```

**Step 4: Validate Results**
```bash
.venv/bin/python -c "
import json
with open('results/results.json') as f:
    data = json.load(f)
print(f'Tokenizers: {data[\"metadata\"][\"num_tokenizers\"]}')
print(f'Languages: {data[\"metadata\"][\"num_languages\"]}')
print(f'Data points: {len(data[\"results\"])}')
en_results = [r for r in data['results'] if r['language'] == 'en']
for r in en_results:
    print(f'  {r[\"tokenizer\"]}: compression={r[\"compression_ratio\"]:.2f}')
    assert 1.0 < r['compression_ratio'] < 20.0
print('Validation passed.')
"
```

**Step 5: Review the Report**
```bash
cat results/report.md
```

- [ ] **Step 2: Commit**

```bash
git add SKILL.md
git commit -m "feat: executable SKILL.md for cross-lingual tokenizer analysis"
```

---

## Task 8: End-to-End Test

Run the full SKILL.md workflow to verify everything works from a clean state.

- [ ] **Step 1: Clean slate**

```bash
rm -rf .venv results/
```

- [ ] **Step 2: Execute SKILL.md steps 1–5**

Run each step from SKILL.md sequentially. Record any failures.

- [ ] **Step 3: Verify outputs**

```bash
ls -la results/
head -40 results/report.md
```

Expected: `results.json` and `report.md` both exist with content.

- [ ] **Step 4: Fix any issues, re-test, commit**

```bash
git add -A
git commit -m "fix: end-to-end test corrections"
```

---

## Task 9: Research Note (LaTeX)

**Files:**
- Create: `research_note/main.tex`

- [ ] **Step 1: Download LaTeX template**

Check https://claw4s.github.io/ for a template link. If none, use a minimal article class.

- [ ] **Step 2: Write main.tex**

Structure (1–4 pages):
1. **Title:** "Cross-Lingual Tokenizer Equity: An Agent-Executable Analysis of Modern LLM Tokenizers"
2. **Authors:** Yun Du and Claw (AI Agent)
3. **Abstract:** 100-word summary
4. **Introduction:** Why tokenizer equity matters (API cost, latency, fairness)
5. **Methodology:** Tatoeba corpus, 14 languages, 4–6 tokenizers, entropy-based metrics
6. **Results:** Key findings (compression ratio table, tax rankings, equity scores)
7. **Discussion:** Which tokenizer is most equitable; CJK measurement challenges; implications
8. **References:** Cite NeurIPS 2023 fairness paper, EMNLP 2024 compression paper, etc.

Fill in actual numerical results from Task 8 output.

- [ ] **Step 3: Compile and verify**

```bash
cd research_note && pdflatex main.tex
```

- [ ] **Step 4: Commit**

```bash
git add research_note/
git commit -m "feat: research note LaTeX document"
```

---

## Task 10: Submit to clawRxiv

- [ ] **Step 1: Register agent**

```bash
curl -X POST http://18.118.210.52/api/auth/register \
  -H "Content-Type: application/json" \
  -d '{"claw_name": "yundu-tokenizer-analysis"}'
```

Save the returned `api_key` securely (do NOT commit it).

- [ ] **Step 2: Submit via Python script**

```bash
.venv/bin/python -c "
import json, urllib.request

# Load content
with open('SKILL.md') as f:
    skill_content = f.read()
with open('results/report.md') as f:
    report_content = f.read()

# Build paper content (markdown for clawRxiv)
content = '''# Cross-Lingual Tokenizer Equity

**Yun Du** and **Claw** (AI Agent)

## Abstract
We present an agent-executable skill for cross-lingual tokenizer analysis...
[Fill in from research note]

''' + report_content

payload = json.dumps({
    'title': 'Cross-Lingual Tokenizer Equity: An Agent-Executable Analysis of Modern LLM Tokenizers',
    'abstract': '[Fill in 100-word abstract]',
    'content': content,
    'tags': ['tokenization', 'cross-lingual', 'information-theory', 'nlp', 'fairness', 'reproducible-research'],
    'human_names': ['Yun Du'],
    'skill_md': skill_content,
}).encode()

req = urllib.request.Request(
    'http://18.118.210.52/api/posts',
    data=payload,
    headers={
        'Content-Type': 'application/json',
        'Authorization': 'Bearer YOUR_API_KEY',
    },
)
resp = urllib.request.urlopen(req)
print(json.loads(resp.read()))
"
```

- [ ] **Step 3: Verify submission**

```bash
curl -s "http://18.118.210.52/api/posts?q=tokenizer+equity" | python3 -m json.tool
```

- [ ] **Step 4: Final commit and push**

```bash
git add -A
git commit -m "feat: complete Claw4S submission — tokenizer analysis"
git push origin main
```
