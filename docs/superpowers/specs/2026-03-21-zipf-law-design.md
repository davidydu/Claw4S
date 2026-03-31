# Design Spec: Zipf's Law Breakdown in Token Distributions

**Date:** 2026-03-21
**Authors:** Yun Du, Lina Ji, Claw
**Status:** Draft

## 1. Motivation

Zipf's law states that in natural language, the frequency of a word is inversely proportional to its rank: f(r) ~ r^(-alpha), where alpha ~ 1.0. This power-law relationship is foundational to information theory and NLP. However, when text is tokenized by modern BPE-based tokenizers (GPT-4o, Mistral, Qwen, etc.), the token frequency distribution may deviate from ideal Zipfian behavior depending on:

- **Corpus type**: Natural language vs. code vs. multilingual text
- **Tokenizer vocabulary size**: Larger vocabularies may produce more Zipfian distributions
- **Language family**: CJK languages, agglutinative languages, etc. may break differently

This submission measures where Zipf's law breaks down across corpus types and tokenizers, and tests whether the degree of Zipfian adherence (the "Zipf exponent") predicts tokenizer compression efficiency.

## 2. Research Questions

1. **How well does Zipf's law hold for BPE token distributions?** Fit the Zipf-Mandelbrot model f(r) = C / (r + q)^alpha to token rank-frequency data and measure goodness of fit (R^2).
2. **Where does Zipf break down?** Identify breakpoints in the rank-frequency curve where the power-law fit deteriorates. Compare head (high-frequency tokens), body, and tail regions.
3. **Do breakpoints differ by corpus type?** Compare natural language (Tatoeba), code (Python/Java snippets from a code dataset), and multilingual text.
4. **Does Zipf exponent predict tokenizer efficiency?** Correlate the fitted alpha with compression ratio (chars/token) from the tokenizer-analysis submission.

## 3. Methodology

### 3.1 Data Sources

- **Natural language**: Tatoeba parallel sentences (reuse from tokenizer-analysis submission). Languages: English, German, French, Spanish, Russian, Chinese, Japanese, Korean, Hindi, Arabic, Turkish, Vietnamese, Finnish, Hebrew. 200 sentences per language, pinned dataset revision.
- **Code**: Python and Java code snippets from `code_search_net` dataset on HuggingFace (pinned revision). 200 samples per language.

### 3.2 Tokenizers

Reuse the same tokenizer configurations from the tokenizer-analysis submission:
- GPT-4o (o200k_base via tiktoken)
- GPT-4 (cl100k_base via tiktoken)
- Mistral-7B (HuggingFace, pinned revision)
- Qwen2.5-7B (HuggingFace, pinned revision)

Skip gated models (Gemma, Llama) to avoid auth requirements.

### 3.3 Analysis Pipeline

1. **Tokenize** each corpus with each tokenizer
2. **Count** token frequencies and rank them
3. **Fit Zipf-Mandelbrot model**: f(r) = C / (r + q)^alpha
   - Use OLS linear regression on log(f) vs log(r + q) to estimate alpha
   - Grid search over q in [0, 0.5, 1, 2, 5, 10] to find best fit
   - Report R^2 goodness of fit
4. **Piecewise fit**: Split rank distribution into 3 regions (head: top 10%, body: 10-90%, tail: bottom 10%) and fit alpha separately in each region
5. **Breakpoint detection**: Identify where the local Zipf exponent deviates > 0.3 from global fit using a sliding window approach
6. **Correlation analysis**: Pearson/Spearman correlation between alpha (Zipf exponent) and compression ratio across (tokenizer, corpus) pairs

### 3.4 Metrics

| Metric | Definition | Purpose |
|--------|-----------|---------|
| Global alpha | Zipf exponent from full rank-frequency fit | Overall Zipfian adherence |
| R^2 | Coefficient of determination of the fit | Goodness of fit |
| Head/body/tail alpha | Piecewise Zipf exponents | Where breakdown occurs |
| Breakpoint rank | Rank where local alpha deviates from global | Characterize breakdown |
| Compression ratio | Characters per token (from tokenizer-analysis) | Tokenizer efficiency |
| Correlation (alpha, compression) | Pearson r between Zipf exponent and compression | Predictive power |

### 3.5 Output Artifacts

- `results/results.json`: All fitted parameters, R^2 values, breakpoints
- `results/report.md`: Summary report with tables and findings
- `results/figures/zipf_fit_{corpus}_{tokenizer}.png`: Log-log rank-frequency plots with fitted lines
- `results/figures/correlation_alpha_compression.png`: Scatter plot of alpha vs compression ratio
- `results/figures/piecewise_exponents.png`: Bar chart of head/body/tail exponents

## 4. Technical Design

### 4.1 Module Structure

```
submissions/zipf-law/
├── SKILL.md
├── run.py                    # Thin orchestrator
├── validate.py               # Result validation
├── requirements.txt          # Pinned dependencies
├── conftest.py               # Pytest path config
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load Tatoeba + code corpora
│   ├── tokenizer_manager.py  # Load tokenizers (reuse patterns)
│   ├── zipf_analysis.py      # Zipf fitting, piecewise analysis, breakpoints
│   ├── plots.py              # Generate matplotlib figures
│   └── report.py             # Generate markdown report
└── tests/
    ├── __init__.py
    ├── test_data_loader.py
    ├── test_zipf_analysis.py
    └── test_plots.py
```

### 4.2 Key Functions

**zipf_analysis.py:**
- `compute_rank_frequency(token_ids: list[int]) -> tuple[np.ndarray, np.ndarray]`: Returns (ranks, frequencies) sorted by frequency descending
- `fit_zipf_mandelbrot(ranks, freqs, q_values=None) -> dict`: Fit alpha, q, R^2, C via OLS on log-log data
- `fit_piecewise_zipf(ranks, freqs) -> dict`: Fit alpha in head/body/tail regions
- `detect_breakpoints(ranks, freqs, window_size=50) -> list[int]`: Find ranks where local alpha changes significantly
- `analyze_corpus(token_ids, label) -> dict`: Full analysis for one (tokenizer, corpus) pair

**data_loader.py:**
- `load_tatoeba_sentences(pairs, max_sentences) -> dict[str, str]`: Same as tokenizer-analysis
- `load_code_samples(languages, max_samples) -> dict[str, str]`: Load code from code_search_net

**plots.py:**
- `plot_zipf_fit(ranks, freqs, fit_params, title, output_path)`: Log-log plot with data + fit line
- `plot_piecewise_comparison(results, output_path)`: Bar chart of regional exponents
- `plot_alpha_compression_correlation(results, output_path)`: Scatter plot

### 4.3 Dependencies

- `tiktoken==0.12.0` (tokenizer)
- `transformers==5.3.0` (HF tokenizers)
- `sentencepiece==0.2.1` (tokenizer dependency)
- `protobuf==7.34.0` (tokenizer dependency)
- `datasets==4.8.3` (data loading)
- `numpy==2.4.3` (numerical computation)
- `scipy==1.17.1` (statistics)
- `matplotlib==3.10.1` (plotting)
- `pytest==9.0.2` (testing)

### 4.4 Runtime Constraints

- Target: < 5 minutes on CPU
- Dataset downloads cached after first run
- Tokenizer downloads cached after first run
- No GPU required, no model inference

## 5. Validation Criteria

- At least 2 tokenizers loaded
- At least 3 corpus types analyzed (2 natural language + 1 code minimum)
- All Zipf fits have R^2 reported
- Global alpha values in plausible range [0.5, 2.0]
- At least 3 figures generated
- Correlation analysis completed with p-values

## 6. Expected Findings (Hypotheses)

1. **Natural language** will show alpha close to 1.0 with high R^2 (> 0.95)
2. **Code** will show lower alpha (flatter distribution) due to repetitive keywords/syntax
3. **CJK languages** may show higher alpha (steeper distribution) due to large character sets mapped to many rare tokens
4. **Larger vocabulary tokenizers** (GPT-4o with 200k vocab) will show more Zipfian distributions than smaller ones
5. **Higher alpha correlates with higher compression ratio** -- more Zipfian = more efficient tokenization

## 7. Limitations

- Corpus sizes are small (200 sentences/samples) -- sufficient for frequency analysis but limited statistical power for tail behavior
- Only BPE-family tokenizers tested (no unigram/WordPiece comparison)
- Code samples limited to Python and Java
- Piecewise breakpoints depend on arbitrary region boundaries
- Linear regression on log-log data has known biases vs. MLE (acceptable for comparative analysis)
