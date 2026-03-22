---
name: grokking-phase-diagrams
description: Train tiny MLPs on modular arithmetic (addition mod 97) and map the grokking phase diagram as a function of weight decay, dataset fraction, and model width. Classifies each training run into four phases (confusion, memorization, grokking, comprehension) and generates heatmap visualizations.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Grokking Phase Diagrams

This skill trains tiny neural networks on modular arithmetic and studies the "grokking" phenomenon — the delayed phase transition from memorization to generalization. It sweeps over weight decay, dataset fraction, and model width to map the full phase diagram.

## Prerequisites

- Requires **Python 3.10+**.
- **No internet access needed** — all data is generated locally (modular arithmetic).
- **No GPU needed** — models are tiny (<20K parameters), trained on CPU.
- Expected runtime: **5-7 minutes** (60 training runs, up to 2500 epochs each, on CPU).
- All commands must be run from the **submission directory** (`submissions/grokking/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import torch, numpy, scipy, matplotlib; print(f'PyTorch {torch.__version__}, NumPy {numpy.__version__} — All imports OK')"
```

Expected output: `PyTorch 2.6.0, NumPy 2.2.4 — All imports OK`

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with all tests passed and exit code 0. Tests cover data generation, model architecture, training loop, phase classification, and sweep logic.

## Step 3: Run the Analysis

Execute the full phase diagram sweep:

```bash
.venv/bin/python run.py
```

Expected: Script runs 60 training experiments (5 weight decays x 4 dataset fractions x 3 hidden dims [16, 32, 64]), prints progress for each run, and exits with code 0. Output files are created in `results/`.

This will:
1. Generate modular addition dataset (all (a,b) pairs for a,b in 0..96, computing (a+b) mod 97)
2. For each hyperparameter combination: train a tiny MLP, log accuracy curves, classify the outcome
3. Generate phase diagram heatmaps showing grokking/memorization/confusion/comprehension regions
4. Generate example training curves illustrating the grokking phenomenon
5. Save results to `results/sweep_results.json`, `results/phase_diagram.json`, and `results/report.md`

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints validation checks (60 runs, all files present, correct grid coverage) and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Review the phase diagram to understand where grokking occurs vs memorization vs comprehension.

The report contains:
- Phase distribution across all 60 runs
- Effect of weight decay on grokking
- Effect of dataset fraction on generalization
- Detailed per-run results table
- Phase diagram heatmaps (one per hidden dimension)
- Example training curves showing the grokking phenomenon

## How to Extend

- **Change the arithmetic operation:** Modify `generate_modular_addition_data()` in `src/data.py` to compute `(a * b) % p` instead of `(a + b) % p`.
- **Change the prime modulus:** Pass a different `p` to `run_sweep()` in `run.py`. Smaller p (e.g., 23) runs faster; larger p may require more epochs.
- **Add sweep dimensions:** Add new hyperparameter lists in `src/sweep.py` (e.g., learning rate, embedding dimension).
- **Change grokking threshold:** Modify `ACC_THRESHOLD` (default 0.95) and `GROKKING_GAP_THRESHOLD` (default 200 epochs) in `src/analysis.py`.
- **Increase training budget:** Adjust `DEFAULT_MAX_EPOCHS` in `src/sweep.py` (default 2000; will increase runtime).
