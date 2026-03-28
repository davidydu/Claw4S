---
name: depth-vs-width-tradeoff
description: Systematically compare deep-narrow vs shallow-wide MLPs under fixed parameter budgets. Sweeps depth (1-8 layers) vs width across sparse parity (compositional) and smooth regression tasks to determine which architectural dimension matters more for different task types.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Depth vs Width Tradeoff in MLPs

This skill runs a controlled experiment comparing deep-narrow vs shallow-wide MLP architectures under fixed parameter budgets on two contrasting tasks.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (all data is generated synthetically).
- Expected runtime: **~3 minutes** on CPU.
- All commands must be run from the **submission directory** (`submissions/depth-width/`).

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

Expected: All tests pass with exit code 0.

## Step 3: Run the Experiments

Execute the full depth-vs-width sweep (24 experiments: 3 budgets x 4 depths x 2 tasks):

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for each experiment and exits with:
```
Done. See results/results.json and results/report.md
```

This runs:
1. **Sparse parity** (compositional): 20-bit inputs, label = XOR of 3 bits. Tests whether depth helps learn compositional boolean functions.
2. **Smooth regression**: 8-dim inputs, target = sin components + pairwise interactions. Tests generalization on smooth functions.

For each task, sweeps 3 parameter budgets (5K, 20K, 50K) x 4 depths (1, 2, 4, 8 hidden layers), adjusting width to keep total parameters constant.

## Step 4: Validate Results

Check that all 24 experiments completed successfully:

```bash
.venv/bin/python validate.py
```

Expected: Prints summary statistics and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Test accuracy/R-squared tables by depth and parameter budget
- Convergence speed tables (epochs to reach threshold)
- Architecture details (width and actual parameter counts)
- Best depth per budget for each task
- Cross-task analysis and key findings

## How to Extend

- **Add a task:** Create a new data generator in `src/tasks.py` returning a dict with `x_train, y_train, x_test, y_test, input_dim, output_dim, task_type, task_name`. Add its hyperparameters to `TASK_HPARAMS` in `src/experiment.py`.
- **Change parameter budgets:** Edit `PARAM_BUDGETS` in `src/experiment.py`.
- **Change depths:** Edit `DEPTHS` in `src/experiment.py`.
- **Change parity difficulty:** Adjust `K_RELEVANT` (higher = harder) and `N_BITS` in `src/experiment.py`.
- **Add a new architecture:** Subclass `nn.Module` in `src/models.py` and modify `run_single_experiment()` in `src/experiment.py`.
- **Add multiple seeds:** Loop over seeds in `run_all_experiments()` to report mean +/- std across runs.
