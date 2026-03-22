---
name: rmt-weight-analysis
description: Analyze eigenvalue spectra of trained MLP weight matrices against the Marchenko-Pastur distribution from Random Matrix Theory. Trains tiny MLPs on modular arithmetic (mod 97) and polynomial regression, then measures how trained weights deviate from random predictions using KS statistics, outlier fractions, and spectral norm ratios.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Random Matrix Theory Analysis of Neural Network Weights

This skill trains tiny MLPs on synthetic tasks and analyzes their weight matrix eigenvalue spectra using Random Matrix Theory (RMT). It compares empirical spectra to the Marchenko-Pastur distribution to quantify how much structure each layer has learned.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (all data is synthetic).
- Expected runtime: **1-3 minutes** on CPU.
- All commands must be run from the **submission directory** (`submissions/rmt/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify installation by running the test suite (Step 2), which will catch any missing dependencies.

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with all tests passed and exit code 0.

## Step 3: Run the Analysis

Execute the full RMT analysis pipeline:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress through 5 stages and exits with code 0. Files created in `results/`:
- `results.json` — raw metrics for all models and layers
- `report.md` — human-readable summary with tables
- `eigenvalue_spectra.png` — eigenvalue histograms vs MP overlay (trained)
- `eigenvalue_spectra_untrained.png` — eigenvalue histograms vs MP overlay (untrained)
- `ks_summary.png` — KS statistics, outlier fractions, and spectral norm ratios

This will:
1. Generate modular addition (mod 97) and polynomial regression datasets
2. Train 8 tiny MLPs (4 hidden dims x 2 tasks) with seed=42
3. Extract weight matrices from each layer (3 per model)
4. Compute eigenvalue spectra of correlation matrices W^T W / M
5. Compare to Marchenko-Pastur theoretical predictions
6. Measure KS statistic, outlier fraction, spectral norm ratio, KL divergence
7. Generate comparison plots and summary report

## Step 4: Validate Results

Check that results are complete and scientifically valid:

```bash
.venv/bin/python validate.py
```

Expected: Prints metric summaries and `Validation passed.` The validator checks:
- All 8 models trained successfully
- All 24 layer analyses (8 models x 3 layers) completed
- Metrics in valid ranges (KS in [0,1], outlier fraction in [0,1])
- Core hypothesis holds: trained models deviate more from MP than untrained

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Training summary (loss, accuracy/MSE per model)
- RMT analysis table for trained models (KS, outlier fraction, spectral norm ratio, KL divergence)
- RMT analysis table for untrained baselines
- Trained vs untrained comparison with delta KS
- Key findings

## How to Extend

- **Add a task:** Create a new data generator in `src/data.py` and add it to the training loop in `run.py`.
- **Change network architecture:** Modify `TinyMLP` in `src/model.py` (e.g., add layers, change activation).
- **Change hidden dimensions:** Edit `HIDDEN_DIMS` list in `run.py`.
- **Add RMT metrics:** Extend `analyze_weight_matrix()` in `src/rmt_analysis.py`.
- **Test on pre-trained models:** Load weights from a saved checkpoint and pass to `analyze_model_weights()`.
