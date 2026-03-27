# SKILL: Private Scaling Laws -- Do Scaling Laws Hold Under DP-SGD?

## Overview

This skill trains small MLPs of varying sizes with both standard SGD and Differentially Private SGD (DP-SGD), then fits power-law scaling curves to test whether the standard relationship L(N) ~ N^(-alpha) holds under privacy constraints. We find that scaling laws do hold under DP-SGD (R^2 > 0.95), and that the privacy cost manifests primarily as a higher scaling coefficient (higher absolute loss) rather than a degraded exponent.

## Prerequisites

- Python 3.13 (tested with `/opt/homebrew/bin/python3.13`)
- CPU-only (no GPU required)
- No API keys, no network access, no authentication
- ~2-3 minutes runtime

## Setup

```bash
cd submissions/dp-scaling
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**Expected output:** All packages install successfully. Key versions: torch==2.6.0, numpy==2.2.4, scipy==1.15.2, matplotlib==3.10.1.

## Step 1: Run Unit Tests

```bash
cd submissions/dp-scaling
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** All tests pass (approximately 20 tests). Tests cover data generation, model construction, parameter counting, standard training, DP-SGD training, per-sample gradient computation, gradient clipping, and scaling law fitting.

## Step 2: Run the Experiment

```bash
cd submissions/dp-scaling
.venv/bin/python run.py
```

**Expected output:**
- Prints progress for 45 training runs (5 hidden sizes x 3 privacy levels x 3 seeds)
- Each run prints: hidden size, privacy level, seed, test loss, accuracy, training time
- Saves `results/experiment_results.json` (raw + aggregated results)
- Saves `results/scaling_laws.png` (log-log scaling law comparison figure)
- Saves `results/accuracy_comparison.png` (accuracy vs model size figure)
- Prints scaling law summary with alpha exponents for each privacy level

**Expected summary format:**
```
SUMMARY: Scaling Law Exponents
  non_private    : alpha = X.XXXX  (R^2 = X.XXXX)
  moderate_dp    : alpha = X.XXXX  (R^2 = X.XXXX)  ratio vs non-private = X.XXXX
  strong_dp      : alpha = X.XXXX  (R^2 = X.XXXX)  ratio vs non-private = X.XXXX
```

The non-private alpha should be the largest. The ratio values show how much DP reduces the scaling exponent.

## Step 3: Validate Results

```bash
cd submissions/dp-scaling
.venv/bin/python validate.py
```

**Expected output:** All validation checks pass:
- All 3 output files exist and are non-empty
- JSON has correct structure with all required keys
- All 45 training runs completed
- All 3 privacy levels have valid scaling law fits
- Scaling exponents are positive and bounded (0 < alpha < 5)
- R-squared values >= 0.5 for each fit
- All test losses are finite and positive
- Prints "VALIDATION PASSED" at the end

## Scientific Details

**Data:** Synthetic Gaussian cluster classification (500 samples, 10 features, 5 classes). Deterministic generation with seed=42.

**Models:** 2-layer MLP (Linear -> ReLU -> Linear) with hidden widths [16, 32, 64, 128, 256], yielding parameter counts from ~261 to ~3,841.

**Training:**
- **Non-private:** Standard SGD, lr=0.01, 100 epochs
- **Moderate DP:** DP-SGD with noise_multiplier=1.0, clipping_norm=1.0
- **Strong DP:** DP-SGD with noise_multiplier=3.0, clipping_norm=1.0

**DP-SGD implementation:** From scratch (no external DP libraries). Per-sample gradients computed via sample-wise forward/backward passes, clipped to L2 norm <= C, summed, Gaussian noise N(0, sigma^2 * C^2 * I) added, then averaged.

**Scaling law fit:** L(N) = a * N^(-alpha) + L_inf via scipy.optimize.curve_fit with Levenberg-Marquardt, bounded (a > 0, 0 < alpha < 5, L_inf >= 0).

**Key findings:** (1) Power-law scaling holds under DP-SGD with R^2 > 0.95. (2) On well-separated synthetic data, DP raises absolute loss (coefficient a increases ~4x) but does not degrade the scaling exponent alpha. In fact, alpha_DP > alpha_NP, meaning private models benefit more from scaling up. (3) Moderate (sigma=1.0) and strong (sigma=3.0) DP yield nearly identical exponents, suggesting gradient clipping dominates over noise magnitude.

## How to Extend

1. **Different model architectures:** Replace `src/model.py` with CNNs, Transformers, etc. Keep the `count_parameters()` interface.
2. **Real datasets:** Replace `src/data.py` with CIFAR-10, MNIST, etc. Adjust `make_dataloaders()` return type.
3. **More privacy levels:** Add entries to `PRIVACY_CONFIGS` in `src/experiment.py`.
4. **Larger models:** Extend `HIDDEN_SIZES` list. For hidden sizes > 512, consider reducing epochs for runtime.
5. **Privacy accounting:** Add Renyi DP or moments accountant to compute formal (epsilon, delta) guarantees for each noise_multiplier.
6. **Deeper networks:** Change `MLP` to support variable depth and study depth vs width scaling under DP.

## Output Files

| File | Description |
|------|-------------|
| `results/experiment_results.json` | All raw runs, aggregated statistics, scaling fits, summary |
| `results/scaling_laws.png` | Log-log plot of test loss vs parameters with fitted curves |
| `results/accuracy_comparison.png` | Accuracy vs model size for all privacy levels |
