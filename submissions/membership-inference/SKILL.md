---
name: membership-inference-scaling
description: Measure how membership inference attack success scales with model size and overfitting gap. Trains tiny MLPs (16-256 hidden units), applies the Shokri et al. (2017) shadow model attack, and analyzes whether attack AUC correlates more strongly with generalization gap or raw model capacity.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Membership Inference Scaling Analysis

This skill runs a membership inference attack experiment measuring how attack
success (AUC) scales with MLP model size and overfitting gap, using the shadow
model approach from Shokri et al. (2017).

## Prerequisites

- Requires **Python 3.10+** (CPU only, no GPU needed).
- Expected runtime: **under 30 seconds** (excluding venv setup).
- All commands must be run from the **submission directory** (`submissions/membership-inference/`).
- No internet access or API keys required (uses synthetic data).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import torch, numpy, scipy, matplotlib, sklearn; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `26 passed` and exit code 0.

## Step 3: Run the Experiment

Execute the full membership inference scaling analysis:

```bash
.venv/bin/python run.py
```

Expected output: The script prints progress for each of 5 model widths (16, 32, 64, 128, 256), showing attack AUC and overfitting gap per width. Final line: `Done in <N>s`. Files `results/results.json` and `results/report.md` are created.

This will:
1. Generate synthetic Gaussian cluster data (500 samples, 10 features, 5 classes)
2. For each of 5 MLP widths (16, 32, 64, 128, 256):
   - Train 3 target models (for variance estimation)
   - Train 3 shadow models per target (same architecture, independent data)
   - Use shadow model predictions to train logistic regression attack classifiers
   - Evaluate attack AUC on target model members vs non-members
3. Compute Pearson correlations: attack AUC vs model size, attack AUC vs overfitting gap
4. Generate 4 plots (PNG) and a summary report

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints per-width AUC and gap summary, correlation analysis, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Review the results table and key findings about whether overfitting gap or model size appears more predictive in this run.

## How to Extend

- **Change model sizes**: Edit `HIDDEN_WIDTHS` in `src/models.py` (list of integers).
- **Change data complexity**: Edit `N_SAMPLES`, `N_FEATURES`, `N_CLASSES` in `src/data.py`.
- **Add more shadow models**: Edit `N_SHADOW_MODELS` in `src/attack.py` (default: 3).
- **Change attack classifier**: Replace `LogisticRegression` in `src/attack.py:train_attack_classifier()` with any sklearn classifier.
- **Use real data**: Replace `generate_gaussian_clusters()` in `src/data.py` with a real dataset loader (ensure same return signature: X, y arrays).
- **Change number of repeats**: Edit `n_repeats` parameter in `run_attack_for_width()` calls in `run.py`.
