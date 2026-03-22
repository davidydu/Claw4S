---
name: double-descent-in-practice
description: Systematically reproduce the double descent phenomenon (Nakkiran et al. 2019, Belkin et al. 2019) using random features models and MLPs on synthetic regression data. Demonstrates model-wise double descent, noise amplification, epoch-wise dynamics, and variance analysis — all on CPU in under 15 seconds.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Double Descent in Practice

This skill reproduces the **double descent phenomenon** — where test error first decreases, then increases sharply at the interpolation threshold, then decreases again — using random ReLU features models and trained MLPs on synthetic data.

## Prerequisites

- Requires **Python 3.10+**. No internet access or GPU needed.
- Expected runtime: **10-15 seconds** on CPU.
- All commands must be run from the **submission directory** (`submissions/double-descent/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import torch, numpy, scipy, matplotlib; print(f'torch={torch.__version__}'); print('All imports OK')"
```

Expected output:
```
torch=2.6.0
All imports OK
```

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass (48 tests). Exit code 0.

## Step 3: Run the Analysis

Execute the full double descent analysis:

```bash
.venv/bin/python run.py
```

Expected: Script completes in under 15 seconds. Prints progress `[1/4]` through `[6/6]` and exits with code 0.

This will:
1. Generate synthetic noisy regression data (n=200, d=20).
2. Sweep random-feature width from 10 to 1000, crossing the interpolation threshold at p=200, for 3 noise levels (sigma=0.1, 0.5, 1.0).
3. Sweep MLP hidden width for comparison.
4. Track MLP test loss over epochs at the interpolation threshold.
5. Repeat with 3 random seeds for variance estimation.
6. Generate 5 publication-quality plots and a summary report.

Output files created in `results/`:
- `results.json` — all raw experimental data.
- `report.md` — summary of findings.
- `model_wise_double_descent.png` — test MSE vs. feature count (3 noise levels).
- `noise_comparison.png` — overlay showing noise amplifies double descent.
- `epoch_wise_double_descent.png` — test MSE vs. training epoch at threshold.
- `mlp_comparison.png` — random features vs. trained MLP side-by-side.
- `variance_bands.png` — mean +/- std across random seeds.

## Step 4: Validate Results

Check that results were produced correctly and double descent was detected:

```bash
.venv/bin/python validate.py
```

Expected output includes:
- Runtime under 180s.
- Peak/min ratio >> 1 for all noise levels (confirming double descent).
- All 5 plot files present.
- Report generated.
- Final line: `Validation passed.`

## Step 5: Review the Report

Read the generated summary:

```bash
cat results/report.md
```

Expected: Markdown report with setup, results tables, and key findings including:
- Model-wise double descent confirmed with peak at p=n=200.
- Peak-to-minimum ratio of several hundred to several thousand.
- Noise amplification effect.
- Benign overfitting in the overparameterized regime.

## How to Extend

### Different data dimensions
In `src/sweep.py`, modify `run_all_sweeps()` config parameters:
- Change `d` for different input dimensions.
- Change `n_train` to shift the interpolation threshold.
- Change `noise_levels` to explore different noise regimes.

### Different model types
- Add new model classes in `src/model.py` (e.g., deeper MLPs, random Fourier features).
- Create corresponding sweep functions in `src/sweep.py`.

### Classification tasks
- Modify `src/data.py` to generate classification data.
- Replace MSE with cross-entropy loss in `src/training.py`.
- Update analysis metrics accordingly.

### Regularization study
- Add weight decay or dropout to the MLP in `src/training.py`.
- Compare double descent curves with/without regularization.

## Key Scientific References

1. Nakkiran et al. (2019) "Deep Double Descent: Where Bigger Models and More Data Hurt" — arXiv:1912.02292
2. Belkin et al. (2019) "Reconciling Modern Machine Learning Practice and the Classical Bias-Variance Trade-off" — PNAS 116(32)
3. Advani & Saxe (2017) "High-dimensional dynamics of generalization error in neural networks" — arXiv:1710.03667
