---
name: activation-sparsity-evolution
description: Track how ReLU activation sparsity evolves during training across model sizes and tasks. Studies whether self-sparsification predicts generalization and whether grokking transitions coincide with sparsity transitions. Trains 8 two-layer MLPs (4 widths x 2 tasks) on CPU with deterministic seeds and reports pooled/task-stratified correlations with bootstrap confidence intervals.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Activation Sparsity Evolution During Training

This skill trains 8 ReLU MLPs (hidden widths 32, 64, 128, 256 on two tasks) and tracks activation sparsity metrics -- dead neuron fraction, zero activation fraction, activation entropy, and mean magnitude -- every 50 epochs over 3000 training epochs. It tests three hypotheses: (1) networks self-sparsify during training, (2) sparsification rate predicts generalization, and (3) grokking transitions in modular arithmetic coincide with sparsity transitions.

## Prerequisites

- **Python 3.10+** available on the system.
- **No GPU required** -- all training runs on CPU.
- **No internet required** -- all data is generated synthetically.
- **Expected runtime:** about 2-3 minutes for `.venv/bin/python run.py` on CPU, plus dependency install time for a fresh `.venv`.
- All commands must be run from the **submission directory** (`submissions/sparsity/`).

## Step 1: Environment Setup

Create a virtual environment and install pinned dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import torch, numpy, scipy, matplotlib; print(f'torch={torch.__version__} numpy={numpy.__version__} scipy={scipy.__version__}'); print('All imports OK')"
```

Expected output: `torch=2.6.0 numpy=2.2.4 scipy=1.15.2` followed by `All imports OK`.

## Step 2: Run Unit Tests

Verify all source modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `33 passed` and exit code 0.

## Step 3: Run the Analysis

Execute the full experiment suite (8 training runs + analysis):

```bash
.venv/bin/python run.py
```

Expected output:
- Phase banners: `[1/4] Generating datasets...`, `[2/4] Running 8 training experiments...`, `[3/4] Computing correlations...`, `[4/4] Analyzing grokking-sparsity transitions...`
- Progress lines for each of 8 experiments, e.g.: `[1/8] modular_addition h=32 lr=0.01 wd=1.0... done (7.0s) dead=0.000 zero_frac=0.475 test_acc=0.580`
- Training summary line: `Total training time: NNN.Ns`
- Plot generation messages: `Saved: results/sparsity_evolution.png` (and 2 more)
- Final line: `[DONE] All results saved to results/`

This will:
1. Generate two synthetic datasets (modular addition mod 97, nonlinear regression)
2. Train 8 two-layer ReLU MLPs (4 hidden widths x 2 tasks) for 3000 epochs each
3. Track dead neuron fraction, zero activation fraction, near-dead fraction, activation entropy, and mean magnitude every 50 epochs
4. Compute Spearman correlations between sparsity metrics and generalization
5. Detect grokking events and check for coincident sparsity transitions
6. Generate three plots and a summary report in `results/`

## Step 4: Validate Results

Check that all results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected output:
```
Experiments: 8 (expected 8)
Correlations: 6 computed
Task-stratified correlation groups: 2
Summaries: 8
Grokking analyses: 4
Hidden widths: [32, 64, 128, 256]
Tasks: ['modular_addition_mod97', 'nonlinear_regression']
  results/report.md: NNNN bytes
  results/sparsity_evolution.png: NNNN bytes
  results/grokking_vs_sparsity.png: NNNN bytes
  results/width_vs_sparsity.png: NNNN bytes

Validation passed.
```

## Step 5: Review the Report

Read the generated analysis report:

```bash
cat results/report.md
```

The report contains:
- Experiment results table (dead neuron fraction, zero fraction, zero fraction change, test accuracy, generalization gap per run)
- Spearman correlation statistics (6 pooled correlations + task-stratified correlations)
- Sample size (`n`) and 95% bootstrap confidence intervals for each correlation
- Grokking-sparsity coincidence analysis for each model width
- Key findings summary with statistical significance
- Limitations section

Generated plots in `results/`:
- `sparsity_evolution.png` -- dead neuron fraction and zero activation fraction over training epochs (2x2 grid)
- `grokking_vs_sparsity.png` -- dual-axis plot of test accuracy and sparsity for modular addition (one panel per width)
- `width_vs_sparsity.png` -- final zero fraction and sparsity change vs hidden width

## Key Scientific Findings

- **Zero fraction strongly predicts pooled generalization**: Spearman rho=-0.857 (p=0.007, 95% bootstrap CI=[-1.000, -0.351]) between final zero fraction and generalization gap across all 8 experiments.
- **Task-dependent sparsification direction**: Regression tasks increase zero fraction during training (+0.024 to +0.052), while modular addition decreases it (-0.050 to -0.127).
- **Within-task uncertainty remains high**: Task-stratified correlations (n=4 per task) have wide confidence intervals, so pooled trends should be treated as preliminary.
- **No grokking observed within 3000 epochs**: None of the four modular-addition widths crossed the grokking threshold; width 256 achieved the highest test accuracy (0.725) without a sharp transition.

## How to Extend

- **Add a hidden width:** Append to `HIDDEN_WIDTHS` in `src/analysis.py`.
- **Change the task:** Add a new data generator in `src/data.py` and a corresponding entry in `run_all_experiments()`.
- **Add a sparsity metric:** Implement in `src/metrics.py` and add to `compute_all_metrics()`.
- **Change the architecture:** Modify `ReLUMLP` in `src/models.py` (e.g., add more layers, change activation).
- **Tune hyperparameters:** Adjust `MOD_ADD_LR`, `MOD_ADD_WD`, `REG_LR`, `REG_WD` in `src/analysis.py`.
- **Vary the seed:** Change `SEED` in `src/analysis.py` or loop over multiple seeds for variance estimation.
- **Increase training epochs:** Change `N_EPOCHS` in `src/analysis.py` to allow more time for grokking (increases runtime proportionally).
