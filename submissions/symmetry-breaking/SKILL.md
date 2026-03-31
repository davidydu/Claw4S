---
name: symmetry-breaking-neural-networks
description: Study how identical incoming hidden weights evolve during neural network training. Initialize 2-layer ReLU MLPs with symmetric first-layer rows, keep the readout layer on seeded Kaiming initialization, add controlled perturbations (epsilon sweep from 0 to 0.1), train on modular addition mod 97 with SGD, and measure symmetry decay via pairwise cosine similarity. Reveals that mini-batch SGD rapidly amplifies this asymmetry, breaking speed scales with network width, and only large perturbations (epsilon = 0.1) yield task-useful representations.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Symmetry Breaking in Neural Network Training

This skill trains 2-layer ReLU MLPs whose incoming hidden weights start symmetric and measures how quickly training drives those rows apart. It sweeps 4 hidden widths x 5 perturbation scales = 20 runs, tracking symmetry decay and learning dynamics.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (all data is generated synthetically).
- Verified runtime: **about 10 minutes** on CPU for the full 20-run sweep (`run.py`) and about **24 seconds** for the unit tests on the March 28, 2026 audit machine.
- All commands must be run from the **submission directory** (`submissions/symmetry-breaking/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
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

Expected: Pytest exits with `25 passed` and exit code 0.

## Step 3: Run the Experiments

Execute the full symmetry-breaking experiment suite:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for 20 runs (4 hidden widths x 5 epsilon values), generates plots, and exits with `[4/4] Saving results to results/`. Verified runtime is about 10 minutes on CPU.

This will:
1. Generate modular addition dataset (a + b mod 97), 80/20 train/test split
2. For each (hidden_dim, epsilon) pair, initialize identical `fc1` rows, keep `fc2` on seeded Kaiming init, and train with SGD
3. Log symmetry metric (mean pairwise cosine similarity of hidden neurons) every 50 epochs
4. Generate 3 plots: symmetry trajectories, accuracy vs epsilon, symmetry heatmap
5. Save results to `results/results.json`, summary to `results/summary.json`, report to `results/report.md`

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints file checks, data point counts, scientific sanity diagnostics, and `Validation passed.`
The validator now also reports:
- chance-level accuracy (`1/modulus`)
- best test accuracy at the highest epsilon
- best-accuracy gain between highest and lowest epsilon

If those signals are too weak (for example, no task-useful high-epsilon run), validation exits non-zero with a clear error message.

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Per-run table: initial/final symmetry, breaking epoch, test accuracy for each (width, epsilon)
- Key findings: mean symmetry and accuracy for zero vs non-zero epsilon
- Breaking speed statistics for substantial perturbations

Verified results from the March 28, 2026 audit run:
- All 20 mini-batch runs reduce the final `fc1` symmetry metric below `0.12`
- Breaking epoch increases with hidden width: `300` (dim 16), `350` (dim 32), `500` (dim 64), `650` (dim 128) for `epsilon <= 1e-4`
- Only `epsilon = 0.1` yields strong task performance at larger widths: width 32 reaches `42.1%` test accuracy, width 64 reaches `52.1%`, width 128 reaches `9.1%`
- Width 64 + epsilon 0.1 achieves `52.1%` test accuracy with `98.8%` train accuracy

Methodological note:
- This code symmetrizes only the incoming hidden-layer weights (`W1`). The readout matrix (`W2`) remains randomly initialized, so interpret the results as the combination of readout asymmetry and mini-batch stochasticity rather than batch noise in isolation.
- In supervisor verification controls, full-batch training at width 16 / epsilon 0 reduced symmetry only to about `0.92` after 2000 epochs, while manually symmetrizing `W2` kept the hidden layer at symmetry `~1.0` for 500 SGD epochs.

## How to Extend

- **Change the task:** Replace the modular addition dataset in `src/data.py` with any classification task. The `generate_modular_addition_data()` function returns `(x_train, y_train, x_test, y_test)`.
- **Add hidden widths:** Pass a different `hidden_dims` list to `run_all_experiments()` in `run.py`.
- **Add epsilon values:** Pass a different `epsilons` list to `run_all_experiments()` in `run.py`.
- **Change the optimizer:** Modify the optimizer in `src/trainer.py` (e.g., replace SGD with Adam) to study how different optimizers interact with symmetry breaking.
- **Multi-layer networks:** Extend `SymmetricMLP` in `src/model.py` to add more hidden layers and track per-layer symmetry.
- **Different symmetric inits:** Modify `_symmetric_init()` in `src/model.py` to try different base weight values or structured symmetries.
- **Isolate pure hidden-unit symmetry:** Add an option to symmetrize `fc2` as well, then compare mini-batch and full-batch training to separate readout asymmetry from batch-noise effects.
