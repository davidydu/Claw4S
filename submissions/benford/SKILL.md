---
name: benford-law-neural-networks
description: Analyze whether the leading digits of trained neural network weight values follow Benford's Law. Trains tiny MLPs on modular arithmetic and sine regression, saves weight snapshots across training, and tests conformity using chi-squared and MAD statistics.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Benford's Law in Trained Neural Networks

This skill investigates whether trained neural network weights obey Benford's Law — the empirical observation that leading significant digits in many naturally occurring datasets follow a logarithmic distribution, with digit 1 appearing ~30% of the time.

## Prerequisites

- Requires **Python 3.10+** (tested with 3.13).
- No internet access required (all data is generated synthetically).
- No GPU required (CPU-only PyTorch).
- Expected runtime (default full run): **~2-3 minutes** on a modern machine.
- Optional smoke test runtime (`--quick --skip-plots`): **~5-15 seconds**.
- All commands must be run from the **submission directory** (`submissions/benford/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify installation by running the test suite (Step 2), which will catch any missing dependencies.

## Step 2: Run Unit Tests

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `31 passed` and exit code 0.

## Step 3: (Optional) Smoke Test Fast Path

Run a fast end-to-end check before the full experiment:

```bash
.venv/bin/python run.py --quick --skip-plots
```

Expected: exits with code 0 in seconds, writes `results/results.json` and `results/report.md`, and logs progress every 100 epochs.

## Step 4: Run the Full Analysis

Execute the full Benford's Law analysis:

```bash
.venv/bin/python run.py
```

Expected: Script prints `Resolved config: ...`, periodic progress logs (every 1000 epochs), then `[4/4] Saving results to results/` and exits with code 0. Creates `results/results.json`, `results/report.md`, and 13 figures in `results/figures/`.

This will:
1. Generate modular arithmetic (mod 97) and sine regression datasets
2. Train 4 tiny MLPs (2 tasks x 2 hidden sizes: 64, 128) for 5000 epochs each
3. Save weight snapshots at epochs 0, 100, 500, 1000, 2000, 5000
4. Extract leading digits from all weight values at each snapshot
5. Compare digit distributions to Benford's Law using chi-squared and MAD tests
6. Analyze per-layer conformity differences
7. Generate control distributions (uniform, normal, Kaiming) for comparison
8. Save results and generate report with visualizations

## Step 5: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints metadata (including software versions and quick/full mode), model MAD trajectories, figure/report checks, and `Validation passed.`

## Step 6: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Benford's Law reference distribution
- Per-model training dynamics (MAD, chi-squared, and bootstrap 95% CI over epochs)
- Per-layer analysis at final epoch
- Control distribution comparisons
- Key findings on Benford conformity in trained weights
- Reproducibility metadata (Python/PyTorch/NumPy/SciPy/Matplotlib versions)

## How to Extend

- **Add a task:** Create a new data generator in `src/data.py` returning `(X_train, y_train, X_test, y_test)` tensors. Add a training block in `run.py`.
- **Change model architecture:** Modify `TinyMLP` in `src/model.py` or create a new `nn.Module` subclass.
- **Add statistical tests:** Extend `src/benford_analysis.py` with additional goodness-of-fit tests (e.g., Kolmogorov-Smirnov).
- **Analyze biases:** Change `layer_filter="weight"` to `layer_filter="bias"` in `analyze_snapshot()` calls.
- **Change sweep/config without code edits:** Use CLI flags such as `--epochs`, `--hidden-sizes`, `--snapshot-epochs`, `--controls-n`, `--seed`, and `--skip-plots`.
