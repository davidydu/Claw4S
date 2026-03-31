---
name: dp-noise-calibration-comparison
description: Compare four differential privacy accounting methods (naive composition, advanced composition, Renyi DP, Gaussian DP) for Gaussian mechanism noise calibration. Pure mathematical analysis — no model training required. Computes privacy loss epsilon across a grid of noise multipliers, composition steps, and failure probabilities, then visualizes tightness ratios and method rankings.
allowed-tools: Bash(git *), Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# DP Noise Calibration Comparison

This skill performs a systematic comparison of four differential privacy accounting methods for calibrating Gaussian mechanism noise. It is a pure mathematical analysis — no ML models, no GPUs, no datasets.

## Prerequisites

- Requires **Python 3.10+**.
- Internet is needed once to install dependencies; analysis/validation are pure local CPU math after install.
- Expected runtime: **< 10 seconds** (pure CPU math).
- All commands must be run from the **submission directory** (`submissions/dp-calibration/`).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/dp-calibration/
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
  (`gdp=65`, `naive=7`, `rdp=0`, `advanced=0` on the pinned grid)
- Average tightness ratios per method
  (approximately `naive=10.607`, `advanced=9.929`, `rdp=1.449`,
  `gdp=1.013` on the pinned grid)
- Robust tightness summaries (median + 95th percentile) for each method
- Wins broken down by composition steps (T)
- Reproducibility fingerprint:
  `Results digest (SHA256): 1d93cec82a3e3e76bb62a347d178fc25ca1a609b9329b1843ebe533b21c70217`

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
3. Reproducibility metadata is present and self-consistent:
   - `results_digest` matches recomputed digest
   - runtime package versions match metadata
4. All methods produce finite epsilon for sigma >= 1.0
5. All tightness ratios >= 1.0 (sanity check)
6. Robust summary stats (median/p95 tightness) are present and valid
7. Scientific findings remain stable on pinned grid:
   - wins = `{naive: 7, advanced: 0, rdp: 0, gdp: 65}`
   - digest = `1d93cec82a3e3e76bb62a347d178fc25ca1a609b9329b1843ebe533b21c70217`
8. All 4 visualization files exist

## Optional: Custom-Grid Sweep (Generalization Check)

Run a custom grid without editing source:

```bash
.venv/bin/python run.py --t-values 50,500 --delta-values 1e-4,1e-5 --sigma-values 0.5,1.0,2.0 --output-dir results/custom
.venv/bin/python validate.py --results-path results/custom/results.json
```

Expected behavior:
- Validation still passes.
- Validator reports `Custom grid detected; pinned-grid checks not applied.`
- Figures are generated in `results/custom/` without user warnings.

## Key Scientific Findings

1. **GDP dominates this grid**: Gaussian DP (f-DP) gives the tightest epsilon bound in 65 of 72 configurations and wins every T slice of the pinned sweep.
2. **RDP is a stable runner-up, not a winner here**: Renyi DP never wins outright on this grid, but stays within roughly 1.09-1.98x of GDP and remains much tighter than naive or advanced composition.
3. **Naive only wins in the near-nonprivate corner**: Naive composition is best only in 7 configurations, all at `sigma=0.1`, where every method yields extremely large epsilon.
4. **Advanced composition rarely helps**: It beats naive in only 6 of 72 configurations, all at `sigma=10` and `T>=1000`, and is otherwise close to naive.
5. **Method choice matters more at large T**: The average RDP/GDP gap grows from about 1.24x at `T=10` to about 1.67x at `T=10000`.

## How to Extend

- **Add new accounting methods**: Implement a function with signature `(sigma, T, delta) -> epsilon` and add it to `METHODS` dict in `src/accounting.py`.
- **Change parameter grid without code edits**: Use `run.py` CLI flags:
  `--t-values`, `--delta-values`, `--sigma-values`, `--output-dir`.
- **Research alternative regimes**: Keep pinned baseline in `results/` for reproducibility, and store exploratory runs under separate directories (e.g., `results/custom/`).
- **Add subsampling**: Extend accounting methods to support Poisson subsampling (sampling rate q), which tightens all bounds.
- **Compare with Opacus/dp-accounting**: Validate results against Google's or Meta's DP accounting libraries.
- **Sensitivity analysis**: Vary the sensitivity parameter (currently fixed at 1) to study calibration for different mechanisms.
