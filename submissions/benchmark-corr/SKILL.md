---
name: llm-benchmark-correlation
description: Analyze correlation, redundancy, and dimensionality of 6 LLM benchmarks across 40 models. Computes Pearson/Spearman correlations, PCA, hierarchical clustering, and greedy benchmark selection to show most benchmarks are redundant — 2 principal components explain 97%+ of variance.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# LLM Benchmark Correlation Analysis

This skill analyzes the correlation structure of 6 common LLM benchmarks (ARC-Challenge, HellaSwag, MMLU, WinoGrande, TruthfulQA, GSM8K) across 40 published models spanning 11 families from 70M to 70B parameters. All data is hardcoded from published sources (Open LLM Leaderboard v1, original model papers) — no model inference or downloads required.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (all data is hardcoded).
- Expected runtime: **< 30 seconds**.
- All commands must be run from the **submission directory** (`submissions/benchmark-corr/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy, scipy, matplotlib, sklearn; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify the data and analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `26 passed` and exit code 0.

## Step 3: Run the Analysis

Execute the full benchmark correlation analysis:

```bash
.venv/bin/python run.py
```

Expected: Script prints `[4/4] Saving results to results/` and exits with code 0. Files created:
- `results/results.json` — all numerical results
- `results/report.md` — human-readable summary
- `results/figures/correlation.png` — Pearson/Spearman heatmaps
- `results/figures/pca_variance.png` — explained variance bar + cumulative line
- `results/figures/model_pca.png` — models in PC1-PC2 space, colored by family
- `results/figures/dendrogram.png` — hierarchical clustering of benchmarks
- `results/figures/redundancy.png` — greedy benchmark selection curve

This will:
1. Compute Pearson and Spearman correlation matrices between all benchmark pairs
2. Run PCA to determine how many components explain 90%+ of variance
3. Perform hierarchical clustering (average linkage) on correlation-distance matrix
4. Run greedy forward selection to find minimal benchmark subsets
5. Analyze model family clustering and scale-performance correlations
6. Generate 5 publication-quality figures and a markdown report

## Step 4: Validate Results

Check that results are scientifically valid:

```bash
.venv/bin/python validate.py
```

Expected: 9 checks pass, prints `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Key findings to verify:
- 2 principal components explain 97%+ of variance (confirming high redundancy)
- ARC-Challenge + TruthfulQA alone capture 95%+ of variance
- TruthfulQA is the least redundant benchmark (avg |r| ~ 0.31)
- PC1 correlates strongly with log(params) (r ~ 0.86, p < 1e-12)

## How to Extend

- **Add a model:** Add an entry to `MODEL_INFO` and a corresponding row to `SCORES` in `src/data.py`.
- **Add a benchmark:** Add the name to `BENCHMARKS` and a column to `SCORES` in `src/data.py`.
- **Change clustering method:** Modify `method="average"` in `run_clustering()` in `src/analysis.py`.
- **Change the distance metric:** Modify `dist = 1.0 - np.abs(corr)` in `run_clustering()`.
- **Adjust PCA threshold:** Modify `np.searchsorted(cumvar, 0.90)` in `run_pca()`.
