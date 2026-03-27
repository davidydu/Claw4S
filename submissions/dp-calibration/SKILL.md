---
name: dp-noise-calibration-comparison
description: Compare four differential privacy accounting methods (naive composition, advanced composition, Renyi DP, Gaussian DP) for Gaussian mechanism noise calibration. Pure mathematical analysis — no model training required. Computes privacy loss epsilon across a grid of noise multipliers, composition steps, and failure probabilities, then visualizes tightness ratios and method rankings.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# DP Noise Calibration Comparison

This skill performs a systematic comparison of four differential privacy accounting methods for calibrating Gaussian mechanism noise. It is a pure mathematical analysis — no ML models, no GPUs, no datasets.

## Prerequisites

- Requires **Python 3.10+** and **no internet access** (pure computation, no downloads).
- Expected runtime: **< 10 seconds** (pure CPU math).
- All commands must be run from the **submission directory** (`submissions/dp-calibration/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy, scipy, matplotlib; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify the accounting and analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass. Exit code 0. Tests cover:
- Correctness of each accounting method against known formulas
- Monotonicity (more noise = less epsilon, more steps = more epsilon)
- Method ordering (naive >= advanced >= RDP/GDP)
- Edge cases and invalid inputs
- Full analysis pipeline completeness and reproducibility

## Step 3: Run the Analysis

Execute the full parameter sweep:

```bash
.venv/bin/python run.py
```

Expected output includes:
- Grid size: 4 T values x 3 delta values x 6 sigma values = 72 configurations
- 288 total computations (72 configs x 4 methods)
- Runtime < 10 seconds
- Method win counts showing which method gives tightest bound
- Average tightness ratios per method
- Wins broken down by composition steps (T)

Expected files created in `results/`:
- `results.json` — full structured results
- `epsilon_vs_T.png` — privacy loss vs composition steps
- `tightness_heatmap.png` — tightness ratio heatmaps for all 4 methods
- `method_comparison.png` — bar charts of win counts and avg tightness
- `epsilon_vs_sigma.png` — privacy loss vs noise multiplier

## Step 4: Validate Results

Run the validation script to check completeness and scientific findings:

```bash
.venv/bin/python validate.py
```

Expected output: `PASS: All checks passed`

Validation checks:
1. results.json exists with expected structure
2. All 72 grid points present
3. All methods produce finite epsilon for sigma >= 1.0
4. All tightness ratios >= 1.0 (sanity check)
5. GDP or RDP wins majority of configurations
6. All 4 visualization files exist

## Key Scientific Findings

1. **GDP dominates at large T**: For T >= 1000, Gaussian DP (f-DP) gives the tightest epsilon bounds due to its CLT-based composition.
2. **RDP excels at moderate T**: For T in [10, 1000], Renyi DP with optimized order selection is competitive or best.
3. **Naive composition is 5-50x loose**: Linear composition overestimates privacy loss dramatically, especially at large T.
4. **Advanced composition helps but not enough**: Sublinear scaling is better than naive but still 2-10x looser than RDP/GDP.
5. **Method choice matters more at large T**: The gap between methods grows with composition steps.

## How to Extend

- **Add new accounting methods**: Implement a function with signature `(sigma, T, delta) -> epsilon` and add it to `METHODS` dict in `src/accounting.py`.
- **Change parameter grid**: Modify `T_VALUES`, `DELTA_VALUES`, `SIGMA_VALUES` in `src/analysis.py`.
- **Add subsampling**: Extend accounting methods to support Poisson subsampling (sampling rate q), which tightens all bounds.
- **Compare with Opacus/dp-accounting**: Validate results against Google's or Meta's DP accounting libraries.
- **Sensitivity analysis**: Vary the sensitivity parameter (currently fixed at 1) to study calibration for different mechanisms.
