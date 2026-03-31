---
name: gradient-norm-phase-transitions
description: Train tiny MLPs on modular addition and regression, tracking per-layer gradient L2 norms throughout training. Test whether gradient norm phase transitions predict generalization transitions (grokking onset) before test accuracy does. Sweep 3 dataset fractions x 2 tasks = 6 runs. Compute cross-correlation lag analysis.
allowed-tools: Bash(git *), Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Gradient Norm Phase Transitions Predict Generalization

This skill trains 2-layer MLPs on grokking-prone (modular addition mod 97) and smooth-learning (regression) tasks, tracking per-layer gradient L2 norms at every epoch. It tests whether gradient norm phase transitions precede test accuracy transitions, serving as an early indicator of generalization.

## Prerequisites

- Requires **Python 3.10+** (tested with 3.13). No GPU needed (CPU only).
- No internet access required (all data is generated synthetically).
- Expected runtime: **about 4-6 minutes** on a modern CPU (observed: ~5.2 minutes on Apple Silicon with Python 3.13 / PyTorch 2.6.0).
- All commands must be run from the **submission directory** (`submissions/gradient-norms/`).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/gradient-norms/
```

All subsequent commands assume you are in this directory.

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

Verify the source modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass. Pytest exits with `X passed` and exit code 0. Tests cover data generation, model architecture, training loop, and analysis functions.

## Step 3: Run the Experiment

Execute the full gradient norm phase transition experiment (6 primary runs + 9 variance runs):

```bash
.venv/bin/python run.py
```

Expected: Script prints training progress for each of the 6 primary runs (2 tasks x 3 fractions), then runs multi-seed variance analysis (3 seeds x 3 fractions for modular addition), generates plots, and saves results. Final output includes a summary table showing gradient transition epoch, metric transition epoch, and lag for each run, plus multi-seed lag statistics. Runtime: about 4-6 minutes on CPU (observed: 309.6s on Apple Silicon with Python 3.13 / PyTorch 2.6.0). Exits with code 0.

`results/results.json` now includes reproducibility metadata (timestamp, runtime, Python/platform, library versions, deterministic setting) in addition to run metrics.

Files created:

- `results/results.json` -- structured experiment results
- `results/run_modular_addition_frac*.png` -- per-run gradient norm + accuracy overlay (3 files)
- `results/run_regression_frac*.png` -- per-run gradient norm + R-squared overlay (3 files)
- `results/summary_grid.png` -- all runs side-by-side with normalized signals
- `results/lag_barchart.png` -- bar chart of gradient-to-metric lag per configuration
- `results/weight_norms.png` -- weight norm trajectories

## Step 4: Validate Results

Check that results are complete and scientifically sound:

```bash
.venv/bin/python validate.py
```

Expected: Prints run-by-run summary including transition epochs, lag values, final metrics, and reproducibility metadata. Validation now enforces:
- all modular-addition runs have positive lag (gradient leads),
- all regression control runs have non-positive lag,
- all variance-analysis lags are positive for each fraction,
- required metadata and plots are present.

Ends with `Validation passed.`

## Step 5: Review Results

Inspect the summary table in the JSON output:

```bash
cat results/results.json
```

Key things to look for:
- **lag_epochs**: positive values mean gradient norm transition PRECEDES the metric transition (supports the thesis)
- **gnorm_transition_epoch vs metric_transition_epoch**: the gap indicates how far ahead gradient norms signal generalization
- **per_layer**: shows which layer's gradients transition first
- **pearson_r / pearson_p**: correlation between gradient norm trajectory and test metric

Review the generated plots to visualize the phase transitions and lag analysis.

## How to Extend

- **Change the task**: Add a new dataset function in `src/data.py` following the same dict interface (`x_train`, `y_train`, `x_test`, `y_test`, `input_dim`, `output_dim`, `task_name`).
- **Change the model**: Modify `src/models.py` to add more layers. Update `get_layer_names()` to include all parameterized layers.
- **Change hyperparameters**: Edit the configuration block at the top of `run.py` (fractions, hidden dim, learning rate, weight decay, epochs).
- **Add metrics**: Extend `src/trainer.py` to track additional quantities (e.g., Hessian eigenvalues, loss landscape curvature).
- **Change the modulus**: Pass a different `modulus` to `make_modular_addition_dataset()` in `run.py`. Larger primes increase task difficulty.
- **Add statistical variance**: Run multiple seeds by looping over seeds in `run.py` and aggregating lag statistics.
