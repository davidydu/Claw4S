---
name: optimizer-grokking-landscape
description: Map the grokking landscape across optimizers (SGD, SGD+momentum, Adam, AdamW) on modular arithmetic (addition mod 97). Sweeps optimizer x learning_rate x weight_decay (36 configs, 750 epochs each) to identify delayed grokking, direct generalization, memorization, and failure modes. Produces heatmaps, training curves, and a summary report.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Optimizer Grokking Landscape

This skill reproduces the grokking phenomenon (Power et al., 2022) and maps which optimizers reliably grok on modular addition mod 97. It sweeps 4 optimizers x 3 learning rates x 3 weight decays = 36 configurations.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (all data is generated synthetically).
- Expected runtime: **4-15 minutes** (CPU only, no GPU required). Runtime depends on CPU speed and machine load.
- All commands must be run from the **submission directory** (`submissions/optimizer-grokking/`).

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

Verify modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass (exit code 0). You should see output like `X passed` where X >= 15.

## Step 3: Run the Experiment

Execute the full optimizer sweep:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for each of 36 runs and exits with code 0. Creates four output files in `results/`:
- `sweep_results.json` — raw data for all 36 runs with per-epoch metrics
- `grokking_heatmap.png` — heatmap showing delayed grokking/direct generalization/memorization/failure per config
- `training_curves.png` — representative train/test accuracy curves
- `report.md` — Markdown summary with outcome counts and grokking delays

Progress output looks like:
```
[1/36] sgd lr=0.1 wd=0.0 ...
        -> failure (train=0.025, test=0.002) [8s elapsed]
...
[36/36] adamw lr=0.01 wd=0.1 ...
        -> grokking (train=1.000, test=1.000) [240s elapsed]
Sweep complete: 36 runs in 240s
```

## Step 4: Validate Results

Check all outputs were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints metadata summary, outcome distribution, and `Validation passed.`

## Step 5: Review the Report

Read the generated summary:

```bash
cat results/report.md
```

The report contains:
- Experimental setup (prime, model, split, hyperparameters)
- Outcome summary table per optimizer (grokking/direct generalization/memorization/failure counts)
- Grokking delay statistics (logged epochs from memorization to delayed generalization)
- Detailed per-run results table
- Key findings

## How to Extend

- **Add an optimizer:** Add a branch to `make_optimizer()` in `src/train.py` and append the name to `OPTIMIZERS` in `src/sweep.py`.
- **Change the task:** Modify `generate_all_pairs()` in `src/data.py` (e.g., multiplication mod p).
- **Change the model:** Modify `ModularMLP` in `src/model.py` (e.g., add layers, change dimensions).
- **Add hyperparameters:** Extend `LEARNING_RATES` or `WEIGHT_DECAYS` in `src/sweep.py`.
- **Increase epochs:** Change `MAX_EPOCHS` in `src/sweep.py` (may increase runtime).
