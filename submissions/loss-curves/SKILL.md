---
name: loss-curve-universality
description: Fit parameterized functions (power law, exponential, stretched exponential, log-power) to training loss curves of tiny MLPs across 4 tasks and 3 model sizes. Tests whether training curves follow universal functional forms with task-dependent exponents using AIC/BIC model selection.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Loss Curve Universality Analysis

This skill trains tiny MLPs on 4 tasks (modular addition mod 97, modular multiplication mod 97, regression, classification) at 3 model sizes (hidden=32, 64, 128), records per-epoch training loss curves, and fits 4 parameterized functional forms to each curve. It tests whether training curves follow universal functional forms with task-dependent exponents.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed; all data is generated synthetically.
- Expected runtime: **3-7 minutes** on CPU-only machines. The modular arithmetic runs are the slowest, and heavily shared machines can take longer.
- All commands must be run from the **submission directory** (`submissions/loss-curves/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import torch, numpy, scipy, matplotlib; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify all analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with all tests passed (20+ tests) and exit code 0.

## Step 3: Run the Analysis

Execute the full loss curve universality analysis:

```bash
.venv/bin/python run.py
```

This will:
1. Train 12 MLP models (4 tasks x 3 hidden sizes) for 1500 epochs each
2. Fit 4 functional forms (power law, exponential, stretched exponential, log-power) to each loss curve
3. Compute AIC/BIC for model selection
4. Analyze universality of best-fit forms and exponent distributions
5. Generate plots and save results

Expected: Script prints progress for each of 12 runs, with the longest pauses during the modular arithmetic tasks, saves results to `results/`, and prints a summary report. Exit code 0. Files created:
- `results/results.json` -- compact results with fits and universality analysis
- `results/full_curves.json` -- full per-epoch loss data for all 12 runs
- `results/report.txt` -- human-readable summary report
- `results/loss_curves_with_fits.png` -- 4x3 grid of loss curves with fitted functions
- `results/aic_comparison.png` -- AIC comparison bar chart by task
- `results/exponent_distributions.png` -- exponent distributions grouped by task

## Step 4: Validate Results

Check that all results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints run counts, task details, majority best-fit form, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.txt
```

The report contains:
- Configuration summary (tasks, hidden sizes, epochs)
- Universality summary: majority best-fit form and fraction
- Best-fit form counts across all 12 runs
- Best form per task
- Per-run table: task, hidden size, params, final loss, best form, AIC, BIC
- Key exponent statistics (mean, std, min, max) per functional form

## How to Extend

- **Add a task:** Add a `make_*_data()` function and entry in `TASK_REGISTRY` in `src/tasks.py`.
- **Add a functional form:** Add an entry to `FUNCTIONAL_FORMS` in `src/curve_fitting.py` with the function, initial guess, bounds, and parameter names.
- **Change model architecture:** Modify `src/models.py` and `build_model()`.
- **Change training hyperparameters:** Modify `N_EPOCHS`, `lr`, `batch_size` in `src/trainer.py` or `src/analysis.py`.
- **Add a hidden size:** Append to `HIDDEN_SIZES` in `src/analysis.py`.
