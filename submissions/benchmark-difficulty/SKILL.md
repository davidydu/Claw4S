---
name: benchmark-difficulty-prediction
description: Predict benchmark question difficulty for LLMs using structural and information-theoretic features alone, without running any LLM. Analyzes ARC-Challenge questions with IRT difficulty scores from Easy2Hard-Bench (NeurIPS 2024), extracts 12 text features, and trains a Random Forest model to test whether surface-level question properties can predict LLM performance.
allowed-tools: Bash(git *), Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Benchmark Difficulty Prediction from Structural Features

This skill analyzes whether structural features of multiple-choice benchmark questions can predict which questions are hard for LLMs, without running any LLM.

## Prerequisites

- Requires **Python 3.10+** and **internet access** (for downloading the Easy2Hard-Bench dataset from HuggingFace, ~2 MB text data).
- Expected runtime: **< 1 minute** on CPU.
- All commands must be run from the **submission directory** (`submissions/benchmark-difficulty/`).
- No GPU, API keys, or model weights required.

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/benchmark-difficulty/
```

All subsequent commands assume you are in this directory.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Expected: Step 2 (pytest) will verify all imports.

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with all tests passed and exit code 0.

## Step 3: Run the Analysis

Execute the full benchmark difficulty prediction pipeline:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for 5 stages and exits with code 0. Files are created in `results/`.

Optional reproducibility/debug flags:

```bash
.venv/bin/python run.py --seed 42 --permutations 200 --output-dir results
.venv/bin/python run.py --use-hardcoded
```

This will:
1. Download 1172 ARC-Challenge questions with IRT difficulty scores from Easy2Hard-Bench (falls back to hardcoded sample of 98 questions if download fails)
2. Extract 12 structural features from each question (no LLM needed)
3. Compute Spearman correlations between each feature and IRT difficulty
4. Train a Random Forest regressor and cross-validate with 5 folds
5. Compare against a dummy mean-prediction baseline on the same folds
6. Run a permutation test on out-of-fold Spearman correlation
7. Generate figures and a summary report
8. Save all results to `results/` with dataset provenance metadata

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints feature correlations, model metrics, baseline metrics,
permutation significance, provenance metadata, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Model performance table (R-squared, MAE, Spearman rho, with cross-validation)
- Baseline comparison table (Random Forest vs dummy mean predictor)
- Permutation significance result for out-of-fold Spearman
- Feature correlations table ranked by absolute Spearman rho
- Feature importance ranking from the Random Forest model
- Key findings and interpretation
- Limitations section
- Reproducibility metadata (dataset/config/split/revision/source)

## Expected Key Findings

- **Negation count** has the strongest Spearman correlation with difficulty (~0.07, p < 0.05)
- **Cross-validated Spearman rho is weak** (~0.13), indicating structural features alone are **insufficient** to predict LLM difficulty
- **Dummy baseline comparison** shows only modest uplift from structural features
- **Permutation testing** can show statistically non-zero rank signal, but practical predictive power remains weak
- This supports the conclusion that LLM difficulty is primarily determined by semantic reasoning demands, not surface-level question properties

## How to Extend

- **Add features:** Define new feature functions in `src/features.py` and add the name to `FEATURE_NAMES`.
- **Use a different benchmark:** Modify `src/data.py` to load MMLU, HellaSwag, or another dataset with difficulty labels.
- **Try a different model:** Replace `RandomForestRegressor` in `src/analysis.py` with gradient boosting, SVM, or a neural network.
- **Stress-test statistical rigor:** Increase `--permutations` to tighten the permutation p-value estimate.
- **Add per-question LLM predictions:** Extend the feature matrix with model-specific features (e.g., perplexity) to compare structural vs. model-aware prediction.
- **Analyze difficulty by subject:** Group ARC questions by topic and compare feature distributions across subjects.
