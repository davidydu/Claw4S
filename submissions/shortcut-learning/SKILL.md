---
name: shortcut-learning-detection
description: Detect and quantify shortcut learning in neural networks. Constructs synthetic data with a spurious shortcut feature perfectly correlated with labels in training but absent at test time. Trains 2-layer MLPs across hidden widths [32, 64, 128] and weight decay [0, 0.001, 0.01, 0.1, 1.0] (45 total runs), measuring shortcut reliance via feature ablation.
allowed-tools: Bash(git *), Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Shortcut Learning Detection

This skill trains neural networks on synthetic data with a spurious shortcut feature, measures their reliance on the shortcut via feature ablation, and tests whether L2 regularization (weight decay) reduces shortcut dependence.

## Prerequisites

- Requires **Python 3.10+** (no GPU needed, CPU only).
- Expected runtime: **1-3 minutes**.
- All commands must be run from the **submission directory** (`submissions/shortcut-learning/`).
- No internet access needed (all data is synthetically generated).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/shortcut-learning/
```

All subsequent commands assume you are in this directory.

## Step 1: Environment Setup

Create a virtual environment and install pinned dependencies:

```bash
rm -rf .venv results
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import torch, numpy, scipy, matplotlib; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify all modules work correctly before running the experiment:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `22 passed` and exit code 0. Tests cover data generation, model construction, training, experiment logic, report wording, and strict results validation.

## Step 3: Run the Experiment

Execute the full 45-configuration sweep (3 hidden widths x 5 weight decays x 3 seeds):

```bash
.venv/bin/python run.py
```

Expected output: Progress log for each of 45 runs, then `[4/4] Saving results to results/`. Creates:
- `results/results.json` — raw and aggregated results
- `results/report.md` — formatted summary with findings table

Each run prints its test accuracy (without shortcut) and shortcut reliance.

## Step 4: Validate Results

Check that results are complete and scientifically sound:

```bash
.venv/bin/python validate.py
```

Expected output:
```
Total configurations: 45
Individual runs: 45
Aggregate entries: 15
...
Validation passed.
```

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report includes a table of all 15 aggregate configurations with mean and standard deviation across seeds, plus key findings about shortcut reliance and regularization effects.

## Key Metrics

| Metric | Definition |
|--------|-----------|
| **Train Acc** | Accuracy on training data (shortcut present) |
| **Test Acc (w/ shortcut)** | Test accuracy with shortcut still correlated |
| **Test Acc (w/o shortcut)** | Test accuracy with shortcut randomized |
| **Shortcut Reliance** | `test_acc_with - test_acc_without` (higher = more dependent on shortcut) |

## Expected Scientific Findings

1. Without regularization, models show significant shortcut reliance (accuracy drops when shortcut is removed).
2. Mild weight decay (`0.001`, `0.01`) does little, while stronger weight decay (`0.1`) can reduce shortcut reliance.
3. Extremely strong weight decay (`1.0`) can drive reliance to zero by preventing learning entirely, so shortcut reliance must be interpreted alongside train/test accuracy.
4. The qualitative pattern is similar across model widths (32, 64, 128 hidden units).

## How to Extend

- **More features:** Change `N_GENUINE` in `src/experiment.py` (default: 10).
- **More regularizers:** Add values to `WEIGHT_DECAYS` list in `src/experiment.py`.
- **Different architectures:** Modify `ShortcutMLP` in `src/model.py` (e.g., add layers, use dropout).
- **Real datasets:** Replace `generate_dataset()` in `src/data.py` with a loader for Waterbirds, CelebA, or other spurious-correlation benchmarks.
- **Other mitigations:** Implement group DRO, JTT, or SUBG in `src/train.py` alongside weight decay.
