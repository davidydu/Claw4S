# Adversarial Transferability Phase Diagram

## Overview

Map how adversarial example transferability between neural networks depends on the source-target capacity ratio. Train pairs of MLPs with varying widths on synthetic Gaussian-cluster data, generate FGSM adversarial examples on each source model, and measure what fraction successfully fool each target model. Produces a 4x4 "phase diagram" of transfer rates, capacity-ratio analysis, and depth-mismatch comparison.

## Prerequisites

- Python 3.13 available as `python3`
- ~200 MB disk space for venv
- CPU only; no GPU required
- Runtime: ~20-25 seconds total on verified CPU runs

## Step 1: Create virtual environment

```bash
cd submissions/transferability
python3 -m venv .venv
```

**Expected output:** `.venv/` directory created (no console output).

## Step 2: Install dependencies

```bash
.venv/bin/pip install -r requirements.txt
```

**Expected output:** Successfully installed torch==2.6.0, numpy==2.2.4, scipy==1.15.2, matplotlib==3.10.1, pytest==8.3.5 (and their transitive deps).

**Pinned versions:**
| Package | Version |
|---------|---------|
| torch | 2.6.0 |
| numpy | 2.2.4 |
| scipy | 1.15.2 |
| matplotlib | 3.10.1 |
| pytest | 8.3.5 |

## Step 3: Run unit tests

```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** 19 tests pass:
- `tests/test_data.py` — 5 tests (dataset shape, types, classes, reproducibility, seed variation)
- `tests/test_models.py` — 5 tests (forward shape, 4-layer forward, param count, width/depth effects)
- `tests/test_adversarial.py` — 6 tests (FGSM perturbation, magnitude bounds, clean accuracy, transfer rate bounds/keys, self-transfer)
- `tests/test_experiment.py` — 3 tests (summary structure, summary values, cross-depth model reuse)

## Step 4: Run full experiment

```bash
.venv/bin/python run.py
```

**Expected output:** Creates `results/` directory with:
- `transfer_results.json` — raw data for all 96 evaluations plus summary statistics
- `transfer_heatmap.png` — 4x4 heatmap of mean transfer rates
- `transfer_by_ratio.png` — transfer rate vs capacity ratio with error bars
- `depth_comparison.png` — same-depth vs cross-depth bar chart

**Expected console output:**
```
Same-arch runs: 48
Cross-depth runs: 48
Runtime: ~20-25s
Diagonal (same-width) mean transfer: 1.0
Off-diagonal mean transfer: ~0.86
```

**What the experiment does:**
1. Generates synthetic 5-class Gaussian cluster data (500 samples, 10 features)
2. Trains 2-layer MLPs with widths [32, 64, 128, 256] (4 models per seed, 3 seeds)
3. For each source-target pair (16 pairs): generates FGSM adversarial examples (epsilon=0.3) on the source, tests transfer to the target
4. Repeats with cross-depth pairs: 2-layer source, 4-layer target
5. Computes summary statistics by capacity ratio (target_width / source_width)

## Step 5: Validate results

```bash
.venv/bin/python validate.py
```

**Expected output:** All checks PASS:
- 48 same-arch results, 48 cross-depth results
- All transfer rates in [0, 1]
- All summary statistics present
- 3 plot files exist and are non-trivial
- Clean accuracies above chance
- FGSM produced adversarial examples

## Key Findings

1. **Self-transfer is perfect:** Transfer rate = 1.0 when source = target (diagonal of heatmap).
2. **Capacity ratio governs transferability:** Transfer rate decreases monotonically as the capacity ratio diverges from 1.0. At ratio 1.0: 100%; at ratio 8.0: ~75%.
3. **Asymmetry:** Small-to-large transfer (ratio > 1) degrades faster than large-to-small (ratio < 1).
4. **Depth mismatch reduces transfer:** Same-width cross-depth transfer (~76.5%) is lower than same-width same-depth transfer (100%), a drop of about 23.5 percentage points.

## How to Extend

### Different epsilon values
In `src/experiment.py`, change the `EPSILON` constant or modify `run_full_experiment()` to sweep over a list of epsilon values.

### Different architectures
Replace `MLP` in `src/models.py` with any `nn.Module` (e.g., CNN, Transformer). The `fgsm_attack` and `compute_transfer_rate` functions are architecture-agnostic.

### Real datasets
Replace `make_gaussian_clusters` in `src/data.py` with any `TensorDataset`. The pipeline handles arbitrary `(X, y)` pairs.

### More capacity metrics
Add model complexity measures (e.g., spectral norm, effective rank) to `src/experiment.py` alongside the current width-ratio metric.
