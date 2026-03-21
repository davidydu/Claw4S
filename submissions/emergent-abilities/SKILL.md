---
name: emergent-abilities-analysis
description: Re-analyze published BIG-Bench and MMLU benchmark data to test whether emergent abilities in LLMs are genuine phase transitions or metric artifacts (Schaeffer et al. 2023). Compares discontinuous (exact match) vs. continuous (partial credit) metrics and computes Metric Sensitivity Index for 8 tasks across GPT-3, LaMDA, and PaLM model families.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Emergent Abilities Analysis: Mirage or Real?

This skill re-analyzes published LLM benchmark data to test the claim by Schaeffer et al. (2023) that emergent abilities are metric artifacts rather than genuine capability phase transitions.

## Prerequisites

- Requires **Python 3.10+** (no GPU, no API keys, no internet access needed after setup).
- Expected runtime: **under 1 minute** on any modern CPU.
- All commands must be run from the **submission directory** (`submissions/emergent-abilities/`).
- All benchmark data is hardcoded from published papers -- no model downloads required.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy, scipy, matplotlib; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `48 passed` and exit code 0.

## Step 3: Run the Analysis

Execute the full emergent abilities analysis:

```bash
.venv/bin/python run.py
```

Expected: Script prints `[4/4] Saving results to results/` and exits with code 0.

This will:
1. Analyze 8 BIG-Bench tasks across GPT-3, LaMDA, and PaLM model families
2. Compare discontinuous (exact match) vs. continuous (partial credit) metrics
3. Compute Metric Sensitivity Index (MSI) for each task
4. Generate synthetic demonstration of the metric artifact mechanism
5. Analyze MMLU scaling across 13 models from 5 families
6. Generate 6 publication-quality figures
7. Save results to `results/results.json` and `results/report.md`

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected output:
```
BIG-Bench tasks analyzed: 8
Tasks with nonlinearity scores: 8
  Likely artifacts (MSI > 2): 7
  Possibly genuine (MSI <= 2): 1
Synthetic demo points: 20
MMLU models analyzed: 13
Report length: ~9400 characters
Figures generated: 6

Validation passed.
```

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Metric Sensitivity Index table for all 8 BIG-Bench tasks
- Synthetic demonstration showing how p -> p^n creates apparent emergence
- MMLU scaling analysis across model families
- Detailed metric comparison tables (exact match vs. partial credit)
- Interpretation and limitations

## Key Scientific Findings

1. **7 of 8 tasks show MSI > 2**: Most apparent emergence is a metric artifact
2. **Synthetic demo confirms mechanism**: Linear per-token improvement creates sharp phase transition under exact-match scoring
3. **MMLU scales smoothly**: Multiple-choice accuracy (more continuous) shows relatively smooth scaling with model size

## How to Extend

- **Add a task**: Add entries to `BIGBENCH_TASKS` and `_BIGBENCH_DATA` in `src/data.py`.
- **Add a model family**: Add entries to `_BIGBENCH_DATA` or `MMLU_DATA` in `src/data.py`.
- **Change the MSI threshold**: Modify the `> 2.0` threshold in `src/analysis.py` and `src/report.py`.
- **Add a new metric**: Implement in `src/metrics.py`, then add to `compute_metric_comparison()` in `src/analysis.py`.
- **Change answer length**: Modify the `n_tokens` field in `BIGBENCH_TASKS` in `src/data.py`.
