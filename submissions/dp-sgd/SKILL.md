# DP-SGD Privacy-Utility Tradeoff

**Skill name:** dp-sgd-privacy-utility
**Authors:** Yun Du, Lina Ji, Claw

## Description

Implements differentially private stochastic gradient descent (DP-SGD) from
scratch — no opacus or external DP libraries — and sweeps noise multiplier
and clipping norm to map the privacy-utility tradeoff. Tests whether there is
a collapse-level "privacy cliff" below which model utility collapses, or
whether clipping dominates the observed degradation on this synthetic task.

## Prerequisites

- Python 3.13 available as `python3.13`
- ~500 MB disk for PyTorch (CPU-only)
- No GPU required
- No API keys or authentication needed

## Steps

### Step 1: Set up environment

```bash
cd submissions/dp-sgd
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Expected output:** `Successfully installed torch-2.6.0 numpy-2.2.4 scipy-1.15.2 matplotlib-3.10.1 pytest-8.3.5` (and dependencies).

### Step 2: Run unit tests

```bash
cd submissions/dp-sgd
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** All tests pass (43 tests at the time of writing). A few
transitive `matplotlib`/`pyparsing` deprecation warnings may appear under
Python 3.13, but the suite should finish with zero failures. Key tests verify:
- Synthetic data generation (shapes, reproducibility, normalization)
- MLP architecture (output shapes, parameter count, seed control)
- Per-sample gradient computation (correct count, shapes, independence)
- Gradient clipping (norm reduction, small gradients unchanged)
- Noise addition (shapes, zero-noise = mean, variance injection)
- Privacy accounting (monotonicity in sigma, steps, finite values)
- End-to-end DP-SGD training (returns expected keys, accuracy in range)
- Non-private baseline (above-chance accuracy)

### Step 3: Run the experiment

```bash
cd submissions/dp-sgd
.venv/bin/python run.py
```

**Expected output:**
- 3 non-private baseline runs (accuracy ~0.99)
- 63 DP-SGD runs (7 noise levels x 3 clipping norms x 3 seeds)
- Runtime: ~45-60 seconds on CPU
- Saves `results/results.json`, `results/summary.json`
- Generates plots: `results/privacy_utility_curve.png`, `results/utility_gap.png`, `results/clipping_effect.png`

Key output lines:
```
Baseline mean accuracy: 0.9933
Privacy cliff: not detected (no config falls below 50% of baseline)
Safe region starts: epsilon >= 0.87
```

### Step 4: Validate results

```bash
cd submissions/dp-sgd
.venv/bin/python validate.py
```

**Expected output:** All validation checks pass:
- results.json exists with correct structure
- 63 DP runs + 3 baseline runs present
- All accuracies in [0, 1]
- Epsilon monotonically decreases as noise increases
- Baseline accuracy reasonable (>= 0.50)
- Privacy-utility tradeoff confirmed (low-noise > high-noise accuracy)
- No cliff epsilon reported when no configuration collapses below 50% of baseline
- Multiple seeds used (>= 3)
- All plots generated
- Runtime <= 180 seconds

## How to Extend

1. **Different datasets:** Replace `src/data.py:generate_gaussian_clusters` with your data loader. The rest of the pipeline is data-agnostic.

2. **Different models:** Replace `src/model.py:MLP` with any `nn.Module`. Per-sample gradients are computed generically via backprop.

3. **Tighter privacy accounting:** The RDP accountant in `src/dpsgd.py:compute_epsilon_rdp` uses simplified bounds. For tighter guarantees, implement the full Poisson subsampling RDP bound from Mironov et al. (2017).

4. **Additional noise multipliers:** Add values to `NOISE_MULTIPLIERS` in `run.py`.

5. **Larger models:** For models with many parameters, replace the loop-based per-sample gradient computation with `torch.func.vmap` for efficiency.

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `NOISE_MULTIPLIERS` | `[0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]` | Noise scale sigma |
| `CLIPPING_NORMS` | `[0.1, 1.0, 10.0]` | Per-sample gradient clip threshold C |
| `SEEDS` | `[42, 123, 456]` | Random seeds for variance estimation |
| `N_SAMPLES` | `500` | Total dataset size |
| `N_FEATURES` | `10` | Input dimensionality |
| `N_CLASSES` | `5` | Number of Gaussian clusters |
| `N_EPOCHS` | `20` | Training epochs per run |
| `LEARNING_RATE` | `0.1` | SGD learning rate |
| `BATCH_SIZE` | `64` | Mini-batch size |
| `DELTA` | `1e-5` | Privacy parameter delta |
