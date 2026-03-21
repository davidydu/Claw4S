---
name: zipf-law-token-distributions
description: Analyze Zipf's law adherence in BPE token frequency distributions across natural language, code, and multilingual corpora. Fits Zipf-Mandelbrot models, detects power-law breakdowns, and tests whether Zipf exponent predicts tokenizer compression efficiency.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Zipf's Law Breakdown in Token Distributions

This skill analyzes how well Zipf's law holds for BPE token frequency distributions across different corpus types (natural language vs. code) and languages. It identifies where the power-law fit breaks down and tests whether the Zipf exponent predicts tokenizer compression efficiency.

## Prerequisites

- Requires **Python 3.10+** and **internet access** (for dataset and tokenizer downloads).
- Expected runtime: **2-4 minutes** on first run (subsequent runs are faster due to caching).
- All commands must be run from the **submission directory** (`submissions/zipf-law/`).
- No GPU or model inference required. Only tokenizers are loaded.
- Four tokenizers are loaded by default (GPT-4o, GPT-4, Mistral, Qwen2.5). All are publicly accessible without authentication.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import tiktoken, transformers, datasets, numpy, scipy, matplotlib, sentencepiece; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify all analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with all tests passed and exit code 0.

## Step 3: Run the Analysis

Execute the full Zipf analysis pipeline:

```bash
.venv/bin/python run.py
```

Expected: Script prints `Analysis complete.` and exits with code 0. The pipeline will:

1. Load Tatoeba sentences for 7 languages (English, German, French, Chinese, Japanese, Arabic, Finnish)
2. Load CodeSearchNet samples for 2 languages (Python, Java)
3. Load 4 tokenizers (GPT-4o, GPT-4, Mistral, Qwen2.5)
4. Tokenize each corpus with each tokenizer (36 combinations)
5. Fit Zipf-Mandelbrot models: f(r) = C / (r + q)^alpha
6. Compute piecewise exponents (head/body/tail regions)
7. Detect breakpoints where local Zipf exponent changes
8. Compute Pearson and Spearman correlation between alpha and compression ratio
9. Generate 4+ figures in `results/figures/`
10. Save results to `results/results.json` and report to `results/report.md`

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints analysis summary for all 36 (tokenizer, corpus) pairs and `Validation passed.`

Validation checks:
- At least 2 tokenizers loaded
- At least 3 corpora analyzed
- At least 6 analyses completed
- All alpha values in plausible range [0.1, 3.0]
- All R^2 values in [0, 1]
- At least 3 figures generated
- Report file exists and is non-trivial

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Global Zipf-Mandelbrot fit table (alpha, q, R^2, compression) for all 36 combinations
- Piecewise exponent table (head/body/tail alpha) for all combinations
- Summary by corpus type (natural language vs code)
- Correlation analysis (Zipf exponent vs compression ratio)
- Automatically detected key findings
- Limitations of the analysis

## Step 6: Review Figures

Examine the generated figures:

```bash
ls results/figures/
```

Expected figures:
- `zipf_fit_*.png`: Log-log rank-frequency plots with Zipf-Mandelbrot fit lines
- `piecewise_exponents.png`: Grouped bar chart comparing head/body/tail exponents
- `correlation_alpha_compression.png`: Scatter plot of alpha vs compression ratio
- `zipf_overlay.png`: Overlay of multiple rank-frequency distributions

## How to Extend

- **Add a tokenizer:** Add an entry to `TOKENIZER_CONFIGS` in `src/tokenizer_manager.py` with type ("tiktoken" or "hf"), encoding/model, and revision.
- **Add a natural language:** Add a pair to `nl_pairs` in `run.py` (e.g., "en-ko") and to `LANG_NAMES` in `src/data_loader.py`.
- **Add a code language:** Add to the `languages` list in the `load_code_samples()` call in `run.py`. Supported: python, java, javascript, php, ruby, go (CodeSearchNet languages).
- **Change Zipf fitting:** Modify `q_values` or fitting method in `src/zipf_analysis.py`. The `fit_zipf_mandelbrot()` function accepts a custom list of q values for grid search.
- **Change piecewise boundaries:** Modify the `head_end` and `tail_start` calculations in `fit_piecewise_zipf()` in `src/zipf_analysis.py`.
- **Adjust breakpoint sensitivity:** Change `window_size` and `threshold` parameters in `detect_breakpoints()`.
