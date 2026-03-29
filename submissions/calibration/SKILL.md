---
name: calibration-under-distribution-shift
description: Train 2-layer MLPs of varying widths on synthetic Gaussian clusters and measure Expected Calibration Error (ECE), Brier score, and overconfidence gaps on in-distribution vs shifted test sets. Produces a reproducible empirical comparison of how calibration changes with model width under covariate shift.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Calibration Under Distribution Shift

This skill investigates how neural network calibration changes under distribution shift as a function of model capacity. It trains 2-layer MLPs of varying widths (16--256 hidden units) on synthetic Gaussian cluster data and measures Expected Calibration Error (ECE), Brier score, and overconfidence gaps across shift magnitudes from 0 to 4.0.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (all data is synthetic).
- Expected runtime: **1-3 minutes end-to-end** including environment setup.
  The core experiment is CPU-only and typically finishes in seconds
  (15 training runs, 75 width-shift-seed evaluations).
- All commands must be run from the **submission directory** (`submissions/calibration/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import torch, numpy, scipy, matplotlib; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify all analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass (exit code 0). You should see 20+ tests covering data generation, model training, metrics computation, and reproducibility metadata.

## Step 3: Run the Experiment

Execute the full calibration experiment grid (5 widths x 5 shifts x 3 seeds = 75 width-shift-seed evaluations, organized as 15 width-seed training runs):

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for each of 15 (width, seed) training runs, generates 5 PDF plots and a markdown report, saves all results to `results/results.json`, and prints the full report. Final line: `Done. 15 experiments completed in <X>s.`

Output files created in `results/`:
- `results.json` — raw/aggregated experiment data plus reproducibility metadata (Python, torch, numpy, deterministic settings)
- `report.md` — markdown summary with ECE/accuracy/Brier tables and key findings
- `ece_vs_shift.pdf` — main result: ECE vs shift magnitude by model width
- `accuracy_vs_shift.pdf` — accuracy degradation under shift
- `brier_vs_shift.pdf` — Brier score under shift
- `reliability_diagrams.pdf` — per-shift reliability diagrams for the largest model
- `overconfidence_gap.pdf` — confidence-accuracy gap under shift

## Step 4: Validate Results

Check that all results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints experiment metadata, verifies all 15 raw results and 25 aggregated entries exist, validates reproducibility metadata and metric ranges, confirms all 5 plots exist, and prints `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- ECE table: mean and std across seeds for each (width, shift) combination
- Accuracy and Brier score tables
- Key findings on in-distribution calibration, severe-shift miscalibration, and overconfidence
- Overconfidence analysis under shift
- Limitations of the study

Treat the generated report as the empirical source of truth for this submission. Capacity-shift patterns should be read from the measured tables and plots rather than assumed in advance.

## How to Extend

- **Add model widths:** Modify `HIDDEN_WIDTHS` in `src/experiment.py`.
- **Add shift magnitudes:** Modify `SHIFT_MAGNITUDES` in `src/experiment.py`.
- **Change architecture:** Replace `TwoLayerMLP` in `src/models.py` with deeper networks.
- **Change data distribution:** Modify `generate_data()` in `src/data.py` to use different cluster shapes or shift types (e.g., rotation instead of translation).
- **Add calibration methods:** Add temperature scaling or Platt scaling in a new `src/calibration.py` module and compare calibrated vs uncalibrated ECE.
- **Change number of seeds:** Modify `SEEDS` in `src/experiment.py` for more/fewer runs.
- **Change ECE bins:** Modify `N_BINS` in `src/experiment.py`.
