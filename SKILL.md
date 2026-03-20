---
name: cross-lingual-tokenizer-analysis
description: Analyze cross-lingual tokenizer efficiency across modern LLMs. Compares compression ratios, fertility rates, entropy, and cross-lingual tax for GPT-4o, Mistral, Qwen, and other tokenizers across 14 languages using Tatoeba parallel sentences.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Cross-Lingual Tokenizer Analysis

This skill performs an information-theoretic analysis of LLM tokenization across 14 languages, measuring how modern tokenizers "tax" different languages relative to English.

## Prerequisites

- Requires **Python 3.10+** and **internet access** (for dataset and model downloads).
- Expected runtime: **3-5 minutes** on first run (subsequent runs are faster due to caching).
- All commands must be run from the **project root directory**.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import tiktoken, transformers, datasets, numpy, scipy, sentencepiece; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `X passed` and exit code 0.

## Step 3: Run the Analysis

Execute the full cross-lingual tokenizer analysis:

```bash
.venv/bin/python -c "
from src.analysis import run_analysis
from src.report import generate_report

results = run_analysis(max_sentences=200)
report = generate_report(results)
print(report)
"
```

Expected: Script prints `[4/4] Saving results to results/` and exits with code 0. Files `results/results.json` and `results/report.md` are created.

This will:
1. Download Tatoeba parallel sentences (200 per language pair)
2. Load all available tokenizers (GPT-4o, GPT-4, Mistral, Qwen, etc.)
3. Tokenize each language's text with each tokenizer
4. Compute metrics: compression ratio, bits-per-character, fertility, cross-lingual tax, vocabulary utilization
5. Save raw results to `results/results.json`
6. Generate a summary report at `results/report.md`

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python -c "
import json
with open('results/results.json') as f:
    data = json.load(f)
print(f'Tokenizers: {data[\"metadata\"][\"num_tokenizers\"]}')
print(f'Languages: {data[\"metadata\"][\"num_languages\"]}')
print(f'Data points: {len(data[\"results\"])}')
assert data['metadata']['num_languages'] >= 10, f'Expected >= 10 languages, got {data[\"metadata\"][\"num_languages\"]}'
assert data['metadata']['num_tokenizers'] >= 2, f'Expected >= 2 tokenizers, got {data[\"metadata\"][\"num_tokenizers\"]}'
expected_points = data['metadata']['num_languages'] * data['metadata']['num_tokenizers']
assert len(data['results']) == expected_points, f'Expected {expected_points} data points, got {len(data[\"results\"])}'
en_results = [r for r in data['results'] if r['language'] == 'en']
for r in en_results:
    print(f'  {r[\"tokenizer\"]}: compression={r[\"compression_ratio\"]:.2f}')
    assert 1.0 < r['compression_ratio'] < 20.0
print('Validation passed.')
"
```

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Review the equity ranking to identify the most and least equitable tokenizers.

The report contains:
- Compression ratio table (characters per token) for each tokenizer × language
- Cross-lingual tax table (relative to English)
- Token entropy table
- Fertility table (tokens per word)
- Equity ranking summary with average and maximum tax per tokenizer
- Notes on CJK measurement limitations

## How to Extend

- **Add a tokenizer:** Add an entry to `TOKENIZER_CONFIGS` in `src/tokenizer_manager.py`.
- **Add a language:** Add a pair to `DEFAULT_PAIRS` and `LANG_NAMES` in `src/data_loader.py`.
- **Change the corpus:** Modify `load_parallel_sentences()` in `src/data_loader.py` to load a different dataset.
- **Change the baseline language:** Pass a different `baseline_compression` in `src/analysis.py`.
