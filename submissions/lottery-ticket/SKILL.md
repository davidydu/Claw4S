---
name: lottery-tickets-at-birth
description: Reproduce a pruning-at-initialization study on tiny 2-layer ReLU MLPs. Sweeps 8 sparsity levels, 3 pruning strategies, 2 tasks, and 3 seeds on modular arithmetic and regression. In the verified default run, structured pruning is the strongest strategy, while global magnitude pruning collapses early on both tasks.
allowed-tools: Bash(git *), Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Lottery Tickets at Birth

This skill reproduces a pruning-at-initialization study on tiny neural networks. It sweeps 8 sparsity levels, 3 pruning strategies, and 2 tasks with 3 seeds each, then reports which strategies preserve performance.

## Prerequisites

- Requires **Python 3.10+**. No internet access or GPU needed.
- Expected runtime: **2-6 minutes on CPU** (depends on BLAS/threading behavior and host load).
- Verified runtime in this worktree on **March 28, 2026**: **153.4s** on Apple silicon CPU.
- All commands must be run from the **submission directory** (`submissions/lottery-ticket/`).
- Training uses a **10% validation split** from the training set for early stopping, and restores the best validation checkpoint before final test evaluation.
- Experiments enable deterministic PyTorch algorithms and record environment provenance (`python_version`, `torch_version`, `numpy_version`, `platform`, `device`) in `results/results.json`.

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/lottery-ticket/
```

All subsequent commands assume you are in this directory.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
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

Expected: Pytest exits with `22 passed` and exit code 0.

## Step 3: Run the Experiment

Execute the full lottery ticket experiment (144 training runs):

```bash
.venv/bin/python run.py
```

Expected output:
- Prints progress for each of 144 runs: `[1/144] task=modular, strategy=magnitude, sparsity=0%, seed=42 ... test_acc=X.XXXX`
- Prints `[4/4] Generating report...` followed by the full report
- Creates files in `results/`:
  - `results.json` — raw experiment data
  - `summary.csv` — machine-readable aggregated metrics (means/std/95% CI by task/strategy/sparsity)
  - `accuracy_vs_sparsity.png` — main accuracy vs sparsity plot
  - `epochs_vs_sparsity.png` — training epochs vs sparsity plot
  - `report.txt` — summary report with key findings and 95% confidence intervals

Runtime: typically 2-6 minutes on CPU; use the `Runtime: ...s` line in `validate.py` output as your measured reference.

## Step 4: Validate Results

Verify all outputs are complete and scientifically reasonable:

```bash
.venv/bin/python validate.py
```

Expected output: `Validation PASSED. All checks OK.` with exit code 0.

The validator checks:
- All 144 runs completed (8 sparsities x 3 strategies x 2 tasks x 3 seeds)
- Dense baselines have reasonable performance (accuracy > 5%, R^2 > 0.5)
- Results metadata records the validation split used for early stopping and reproducibility provenance
- All plots and reports were generated
- `summary.csv` exists and contains CI columns
- Each configuration has exactly 3 seeds for variance estimation

## Step 5: Review Key Findings

Read the generated report:

```bash
cat results/report.txt
```

Expected findings:
- **Modular arithmetic**: Dense accuracy is only about `0.29`, magnitude and random pruning collapse by `30%` sparsity, while **structured pruning improves accuracy** and peaks near `70%` sparsity (`~0.71` mean test accuracy)
- **Regression**: **Structured pruning is the most robust**, staying above `0.94` test `R^2` through `90%` sparsity; random pruning degrades gradually; magnitude pruning collapses from `50%` sparsity onward
- **Critical sparsity**: In the verified run, magnitude reaches `0%` (modular) / `30%` (regression), random reaches `0%` / `50%`, and structured reaches `90%` on both tasks
- **Interpretation**: In this tiny-network setting, pruning behaves more like architecture/regularization selection than classic magnitude-based ``winning tickets at birth''

## Interpreting Results

### Accuracy vs Sparsity Plot (`results/accuracy_vs_sparsity.png`)
- X-axis: Sparsity percentage (0% = dense, 95% = almost all weights removed)
- Y-axis: Test accuracy (modular) or test R^2 (regression)
- Three lines per task: one per pruning strategy
- Dashed vertical line: critical sparsity for magnitude pruning (included for historical comparison)

### Key Metrics
| Metric | Description |
|--------|-------------|
| Test Accuracy | Fraction of correct predictions on held-out set (modular task) |
| Test R^2 | Coefficient of determination on held-out set (regression task) |
| Critical Sparsity | Highest sparsity maintaining 95% of dense performance |
| Epochs to Convergence | Training steps before early stopping |
| Validation Split | Fraction of the training set reserved for early stopping |

## How to Extend

### Different Model Sizes
Edit `src/experiment.py` and change `HIDDEN_DIM`:
```python
HIDDEN_DIM = 256  # default is 128
```

### Different Tasks
Add new data generators in `src/data.py` and corresponding training functions in `src/train.py`. Register them in `src/experiment.py`.

### Different Pruning Strategies
Add new pruning functions to `src/pruning.py` following the same API:
```python
def my_prune(model: nn.Module, sparsity: float, seed: int = 42) -> dict:
    # Returns {param_name: mask_tensor}
```
Then add the function to `PRUNING_FNS` in `src/experiment.py`.

### More Sparsity Levels
Edit `SPARSITY_LEVELS` in `src/experiment.py`:
```python
SPARSITY_LEVELS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
```

### More Seeds
Edit `SEEDS` in `src/experiment.py`:
```python
SEEDS = [42, 123, 7, 256, 999]
```
