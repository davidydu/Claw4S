---
name: symmetry-breaking-neural-networks
description: Study how symmetry breaks during neural network training. Initialize 2-layer ReLU MLPs with symmetric weights (all neurons identical), add controlled perturbations (epsilon sweep from 0 to 0.1), train on modular addition mod 97 with SGD, and measure symmetry decay via pairwise cosine similarity. Reveals that SGD batch noise breaks symmetry even from perfectly symmetric init, but breaking speed scales with network width, and only large perturbations (epsilon >= 0.1) yield task-useful representations.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Symmetry Breaking in Neural Network Training

This skill trains 2-layer ReLU MLPs from symmetric initialization and measures how SGD batch noise breaks weight symmetry. It sweeps 4 hidden widths x 5 perturbation scales = 20 runs, tracking symmetry decay and learning dynamics.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (all data is generated synthetically).
- Expected runtime: **2-3 minutes** on CPU.
- All commands must be run from the **submission directory** (`submissions/symmetry-breaking/`).

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

Verify all modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: `24 passed` with exit code 0.

## Step 3: Run the Experiments

Execute the full symmetry-breaking experiment suite:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for 20 runs (4 hidden widths x 5 epsilon values), generates plots, and exits with `[4/4] Saving results to results/`. Runtime ~2 minutes on CPU.

This will:
1. Generate modular addition dataset (a + b mod 97), 80/20 train/test split
2. For each (hidden_dim, epsilon) pair, initialize a symmetric MLP and train with SGD
3. Log symmetry metric (mean pairwise cosine similarity of hidden neurons) every 50 epochs
4. Generate 3 plots: symmetry trajectories, accuracy vs epsilon, symmetry heatmap
5. Save results to `results/results.json`, summary to `results/summary.json`, report to `results/report.md`

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints file checks, data point counts, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Per-run table: initial/final symmetry, breaking epoch, test accuracy for each (width, epsilon)
- Key findings: mean symmetry and accuracy for zero vs non-zero epsilon
- Breaking speed statistics for substantial perturbations

Key results to look for:
- All 20 runs break symmetry (final symmetry < 0.12 in all cases)
- Breaking epoch increases with hidden width (300 for dim=16, 650 for dim=128)
- Only epsilon=0.1 achieves meaningful test accuracy (>5%), especially at larger widths
- Width 64 + epsilon 0.1 achieves ~52% test accuracy with ~99% train accuracy

## How to Extend

- **Change the task:** Replace the modular addition dataset in `src/data.py` with any classification task. The `generate_modular_addition_data()` function returns `(x_train, y_train, x_test, y_test)`.
- **Add hidden widths:** Pass a different `hidden_dims` list to `run_all_experiments()` in `run.py`.
- **Add epsilon values:** Pass a different `epsilons` list to `run_all_experiments()` in `run.py`.
- **Change the optimizer:** Modify the optimizer in `src/trainer.py` (e.g., replace SGD with Adam) to study how different optimizers interact with symmetry breaking.
- **Multi-layer networks:** Extend `SymmetricMLP` in `src/model.py` to add more hidden layers and track per-layer symmetry.
- **Different symmetric inits:** Modify `_symmetric_init()` in `src/model.py` to try different base weight values or structured symmetries.
