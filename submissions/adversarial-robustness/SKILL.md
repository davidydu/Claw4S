# Adversarial Robustness Scaling

## Overview

This skill trains 2-layer ReLU MLPs of varying widths (16 to 512 neurons) on two synthetic 2D classification tasks (concentric circles and two moons), generates adversarial examples using FGSM and PGD attacks across an epsilon sweep, and measures how the robustness gap (clean accuracy minus robust accuracy) scales with model capacity. Experiments run across 3 random seeds for statistical variance, totaling 180 individual evaluations.

**Key finding:** In the small-model regime, the robustness gap is largely independent of model capacity. Contrary to the hypothesis that larger models are more vulnerable, the gap remains stable (circles: r = -0.06) or slightly decreases (moons: r = -0.67) with model size, suggesting that adversarial vulnerability in small MLPs is governed by task geometry rather than model capacity.

## Prerequisites

- Python 3.13 at `/opt/homebrew/bin/python3.13`
- ~500 MB disk for PyTorch (CPU-only)
- No GPU required; total runtime ~60-90 seconds on CPU
- No API keys or authentication needed

## Step 1: Set up the virtual environment

```bash
cd submissions/adversarial-robustness
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

**Expected output:** `Successfully installed torch-2.6.0 numpy-2.2.4 scipy-1.15.2 matplotlib-3.10.1 pytest-8.3.5 ...`

## Step 2: Run unit tests

```bash
cd submissions/adversarial-robustness
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** All tests pass (39 tests). Example:
```
tests/test_data.py::TestMakeCircles::test_output_shape PASSED
tests/test_models.py::TestTwoLayerMLP::test_param_count_formula PASSED
tests/test_attacks.py::TestFGSMAttack::test_perturbation_bounded PASSED
tests/test_attacks.py::TestPGDAttack::test_pgd_stronger_than_fgsm PASSED
tests/test_analysis.py::TestComputeSummaryStatistics::test_correlation_computed PASSED
...
====== 39 passed in Xs ======
```

## Step 3: Run the experiment

```bash
cd submissions/adversarial-robustness
.venv/bin/python run.py
```

This runs the full experiment pipeline:
1. For each of 2 datasets (circles, moons) and 3 seeds (42, 123, 7):
   - Generates 2000-sample dataset (1600 train, 400 test, noise=0.15)
   - Trains 6 MLPs (hidden widths: 16, 32, 64, 128, 256, 512) to convergence
   - For each model, generates FGSM and PGD adversarial examples at 5 epsilon values (0.01, 0.05, 0.1, 0.2, 0.5)
2. Computes clean accuracy, robust accuracy, robustness gaps, and cross-seed aggregated statistics
3. Generates plots and saves all 180 experiment results

**Expected output:**
```
======================================================================
Adversarial Robustness Scaling Experiment
======================================================================
Hidden widths: [16, 32, 64, 128, 256, 512]
Epsilons:      [0.01, 0.05, 0.1, 0.2, 0.5]
Seeds:         [42, 123, 7]
Datasets:      ['circles', 'moons']
Total runs:    180

[Dataset: circles] (noise=0.15)
  Seed=42:
    Width=  16 (354 params): XXX epochs, clean_acc=0.95XX
    ...
    Width= 512 (265,218 params): XX epochs, clean_acc=0.95XX
  Seed=123:
    ...
  Seed=7:
    ...

[Dataset: moons] (noise=0.15)
  ...

Total training + evaluation time: ~60-90s

  [CIRCLES] Per-width summary (mean +/- std across 3 seeds):
   Width   Params        Clean         FGSM Gap          PGD Gap
  -----------------------------------------------------------------
      16      354 0.94XX+/-0.0XXX 0.30XX+/-0.0XXX 0.31XX+/-0.0XXX
     ...
     512   265218 0.94XX+/-0.0XXX 0.32XX+/-0.0XXX 0.33XX+/-0.0XXX

  Corr(log params, FGSM gap): ~-0.06
  Corr(log params, PGD gap):  ~-0.16

  [MOONS] Per-width summary (mean +/- std across 3 seeds):
  ...
  Corr(log params, FGSM gap): ~-0.67
  Corr(log params, PGD gap):  ~-0.18

======================================================================
Experiment complete. Results saved to results/
======================================================================
```

**Runtime:** ~60-90 seconds on CPU.

**Generated files:**
| File | Description |
|------|-------------|
| `results/results.json` | All 180 experiment results + aggregated stats + summary |
| `results/clean_vs_robust.png` | Clean vs robust accuracy across model sizes (circles) |
| `results/robustness_gap.png` | Robustness gap vs model size per epsilon (circles) |
| `results/param_scaling.png` | Mean robustness gap vs parameter count (log scale) |

## Step 4: Validate results

```bash
cd submissions/adversarial-robustness
.venv/bin/python validate.py
```

**Expected output:**
```
============================================================
Adversarial Robustness Scaling -- Validation Report
============================================================

PASSED -- all checks passed.

Configuration: 2 datasets, 3 seeds, 180 total experiments
  - 30 per-seed experiments across 6 model sizes
  - Corr(log params, FGSM gap): -0.XXXX
  - Corr(log params, PGD gap):  -0.XXXX
```

Validation checks:
- All 180 experiments present (6 widths x 5 epsilons x 3 seeds x 2 datasets)
- All accuracies in [0, 1]
- Robustness gaps consistent (gap = clean_acc - robust_acc)
- All models achieve >= 80% clean accuracy on both datasets
- PGD at least as strong as FGSM (within tolerance)
- Robust accuracy generally decreases with epsilon
- Cross-seed aggregated results present (60 entries)
- Summary statistics, correlation values, and plots present and non-empty

## How to Extend

### Different datasets
In `run.py`, modify the `DATASETS` list:
```python
DATASETS = [
    {"name": "circles", "noise": 0.15},
    {"name": "moons", "noise": 0.15},
]
```
Add new generators in `src/data.py` following the same pattern.

### Different model sizes
In `src/models.py`, modify the `HIDDEN_WIDTHS` list:
```python
HIDDEN_WIDTHS = [8, 16, 32, 64, 128, 256, 512, 1024]
```

### Different perturbation strengths
In `src/attacks.py`, modify the `EPSILONS` list:
```python
EPSILONS = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
```

### More random seeds
In `run.py`, modify the `SEEDS` list:
```python
SEEDS = [42, 123, 7, 0, 999]
```
Also update `EXPECTED_SEEDS` in `validate.py` to match.

### Stronger PGD attacks
In `run.py`, increase `n_steps` in the `pgd_attack` call:
```python
pgd_acc = evaluate_robust(model, X_test, y_test, pgd_attack, epsilon=eps, n_steps=50)
```

### 3D input features
Add a 3D generator in `src/data.py` and set `input_dim=3` when calling `build_model()`.

## Methodology Notes

- **FGSM** (Goodfellow et al., 2015): Single-step attack. Perturbs inputs by `epsilon * sign(gradient)`.
- **PGD** (Madry et al., 2018): Multi-step iterative attack (10 steps, step_size=epsilon/4). Projects perturbations back into the L-inf epsilon-ball after each step.
- **Robustness gap**: Defined as `clean_accuracy - robust_accuracy`. Positive values indicate adversarial vulnerability.
- All models trained with Adam (lr=1e-3) with early stopping (patience=50 epochs).
- Three random seeds (42, 123, 7) for statistical variance across data generation, model initialization, and training.
- Two synthetic datasets tested: concentric circles (radial decision boundary) and two moons (crescent-shaped boundary).
