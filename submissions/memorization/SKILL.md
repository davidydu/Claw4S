---
name: memorization-capacity-scaling
description: Systematically test how many random labels neural networks of different sizes can memorize (Zhang et al. 2017). Sweep model size to find the interpolation threshold where #params ~ #samples, and measure whether the transition is sharp or gradual.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Memorization Capacity Scaling

This skill reproduces and extends the classic Zhang et al. (2017) memorization experiment. It trains 2-layer MLPs of varying width on synthetic data with random vs. structured labels, measuring the interpolation threshold (parameter count where 100% training accuracy is first achieved) and characterizing whether the transition is sharp or gradual via sigmoid fitting.

## Prerequisites

- Requires **Python 3.10+** (tested with 3.13). No GPU needed — CPU-only PyTorch.
- Expected runtime: **about 5-8 minutes** on a modern laptop for the full 3-seed sweep.
- All commands must be run from the **submission directory** (`submissions/memorization/`).
- No internet access required (synthetic data only).

## Step 0: Clean Previous Artifacts

For a cold reproducibility run, clear prior artifacts:

```bash
rm -rf results
```

Expected: `results/` is absent before starting.

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

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with all tests passed (20+ tests) and exit code 0.

## Step 3: Run the Experiment

Quick smoke run (fast sanity check, optional):

```bash
.venv/bin/python run.py --seeds 42 --hidden-dims 5,10 --max-epochs 200 --no-plots
```

Expected: Script exits with code 0 and writes `results/results.json` + `results/report.md` (plots intentionally skipped). Use this for quick sanity only.

Full reproducibility run (recommended for paper-quality results and required before `validate.py`):

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for 48 training runs (8 hidden widths x 2 label types x 3 seeds), then prints key results and exits with code 0. On a modern laptop this full sweep typically takes about 5-8 minutes. Files are created in `results/`.

This will:
1. Generate synthetic dataset (200 train, 50 test, 20 features, 10 classes)
2. Train MLPs with hidden widths [5, 10, 20, 40, 80, 160, 320, 640] on both random and structured labels
3. Measure training accuracy (memorization) and test accuracy (generalization)
4. Fit sigmoid to train_acc vs log(#params) to measure transition sharpness
5. Detect interpolation threshold (smallest model achieving 99%+ train accuracy)
6. Save seed-42 sweep results plus 3-seed aggregate statistics to `results/results.json`, report to `results/report.md`, figures to `results/figures/`
7. Record reproducibility metadata (`run_metadata`) including dependency versions, timestamps, and exact run configuration

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints experiment summary, run metadata summary, output file sizes, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Results table for each label type (hidden dim, #params, train/test accuracy)
- Interpolation threshold (parameter count at 99% train accuracy)
- Sigmoid fit parameters (threshold, sharpness, R-squared)
- Multi-seed variance summary (mean +/- std across seeds 42, 43, 44)
- Comparative analysis (random vs. structured labels)
- Key findings and limitations

## How to Extend

- **Change dataset size:** `.venv/bin/python run.py --n-train 500 --n-test 100`
- **Change feature dimension/classes:** `.venv/bin/python run.py --d 50 --n-classes 20`
- **Add/remove hidden widths:** `.venv/bin/python run.py --hidden-dims 10,20,40,80,160`
- **Increase statistical power:** `.venv/bin/python run.py --seeds 42,43,44,45,46`
- **Faster debug loop:** `.venv/bin/python run.py --seeds 42 --hidden-dims 5,10 --max-epochs 200 --no-plots`
- **Different optimizer / architecture:** Modify `src/train.py` and/or `src/model.py` for optimizer or network-depth ablations.
- **Real datasets:** Replace `generate_dataset()` in `src/data.py` with a dataset loader (e.g., MNIST/CIFAR).
