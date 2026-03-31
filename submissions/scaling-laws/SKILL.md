---
name: scaling-laws-verification
description: Verify neural scaling laws using published Cerebras-GPT and Pythia data. Fits Kaplan, Chinchilla, and corrected power-law formulations, compares loss scaling (robust) vs task scaling (unreliable), and quantifies extrapolation risk with parametric bootstrap confidence intervals.
allowed-tools: Bash(git *), Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Scaling Laws Verification

This skill performs a statistical verification of neural scaling laws using published data from Cerebras-GPT (7 model sizes) and Pythia (8 model sizes), demonstrating that loss scaling is robust while task-specific scaling is unreliable.

## Prerequisites

- Requires **Python 3.10+** and **no internet access** needed (all data is embedded).
- Expected runtime: **1-3 minutes** (depends on CPU speed; parametric bootstrap with B=500).
- All commands must be run from the **submission directory** (`submissions/scaling-laws/`).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/scaling-laws/
```

All subsequent commands assume you are in this directory.

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

Expected: All tests pass. Integration tests run actual curve fitting, so this step may take 30-60 seconds.

## Step 3: Run the Analysis

Execute the full scaling laws verification:

```bash
.venv/bin/python run.py
```

Expected: Script prints `[1/5]` through `[5/5]` phase banners and the final report. Files `results/results.json` and `results/report.md` are created. Five figures are saved to `results/figures/`:
- `loss_scaling.png`
- `task_scaling.png`
- `residuals.png`
- `model_selection.png`
- `extrapolation.png`

This will:
1. Fit three scaling law formulations (Kaplan, Chinchilla, corrected) to Cerebras-GPT training losses
2. Fit bounded power-law and sigmoid models to 7 downstream task benchmarks
3. Compute cross-metric correlations between loss improvement and task improvement
4. Quantify extrapolation risk by training on small models and predicting large ones
5. Test cross-family transfer from Cerebras-GPT to Pythia benchmarks

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints 7 validation checks (each showing PASS) and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Review the analysis to see which scaling law formulation fits best, which tasks scale poorly, and how extrapolation risk differs between loss and task metrics. The report contains these sections: Loss Scaling, Task Scaling, Cross-Metric Correlation, Extrapolation Risk, Cross-Family Transfer, Methodology, Limitations.

## How to Extend

- **Add a model family:** Add a new dict to `src/data.py` following the existing CEREBRAS_GPT format, then update `src/analysis.py:run_full_analysis()` to include the new family.
- **Add a downstream task:** Add accuracy values to the model dicts in `data.py`. The task analysis auto-discovers all task keys.
- **Add a scaling formulation:** Add a function to `src/scaling_models.py` and register it in the FORMULATIONS dict.
- **Change bootstrap samples:** Adjust `n_bootstrap` in `run.py` (default: 500; increase to 1000 for tighter CIs, ~2x slower).
