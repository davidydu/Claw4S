---
name: lottery-tickets-at-birth
description: Test the lottery ticket hypothesis on tiny networks — can randomly pruned subnetworks at initialization train as well as dense ones? Prunes 2-layer ReLU MLPs at various sparsity levels (0–95%) using magnitude, random, and structured pruning, then trains on modular arithmetic (mod 97) and regression tasks. Compares final test accuracy to find the critical sparsity threshold.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Lottery Tickets at Birth

This skill tests the lottery ticket hypothesis on tiny neural networks: can subnetworks pruned at initialization match the performance of the dense network? It sweeps over 8 sparsity levels, 3 pruning strategies, and 2 tasks with 3 seeds each.

## Prerequisites

- Requires **Python 3.10+**. No internet access or GPU needed.
- Expected runtime: **1-3 minutes** (CPU only, ~144 training runs of small MLPs).
- All commands must be run from the **submission directory** (`submissions/lottery-ticket/`).

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

Expected: Pytest exits with `20 passed` and exit code 0.

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
  - `accuracy_vs_sparsity.png` — main accuracy vs sparsity plot
  - `epochs_vs_sparsity.png` — training epochs vs sparsity plot
  - `report.txt` — summary report with key findings

Runtime: 1-3 minutes on CPU.

## Step 4: Validate Results

Verify all outputs are complete and scientifically reasonable:

```bash
.venv/bin/python validate.py
```

Expected output: `Validation PASSED. All checks OK.` with exit code 0.

The validator checks:
- All 144 runs completed (8 sparsities x 3 strategies x 2 tasks x 3 seeds)
- Dense baselines have reasonable performance (accuracy > 5%, R^2 > 0.5)
- All plots and reports were generated
- Each configuration has exactly 3 seeds for variance estimation

## Step 5: Review Key Findings

Read the generated report:

```bash
cat results/report.txt
```

Expected findings:
- **Modular arithmetic**: Magnitude-pruned networks maintain accuracy up to ~70-80% sparsity, then degrade
- **Regression**: Performance is more robust to pruning, maintaining R^2 up to ~90% sparsity
- **Magnitude > Random > Structured**: Magnitude pruning consistently outperforms random and structured pruning at high sparsity
- **Critical sparsity**: The point where performance drops below 95% of dense baseline varies by task

## Interpreting Results

### Accuracy vs Sparsity Plot (`results/accuracy_vs_sparsity.png`)
- X-axis: Sparsity percentage (0% = dense, 95% = almost all weights removed)
- Y-axis: Test accuracy (modular) or test R^2 (regression)
- Three lines per task: one per pruning strategy
- Dashed vertical line: critical sparsity (95% of dense performance)

### Key Metrics
| Metric | Description |
|--------|-------------|
| Test Accuracy | Fraction of correct predictions on held-out set (modular task) |
| Test R^2 | Coefficient of determination on held-out set (regression task) |
| Critical Sparsity | Highest sparsity maintaining 95% of dense performance |
| Epochs to Convergence | Training steps before early stopping |

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
