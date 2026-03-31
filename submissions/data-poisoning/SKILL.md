# Data Poisoning Sensitivity: Critical Thresholds in Label-Flip Attacks

## Overview

This skill sweeps poison fraction (0%--50%) on 2-layer MLP classifiers trained on synthetic Gaussian cluster data to identify the critical threshold where model accuracy collapses. The experiment tests whether there is a sharp phase transition or gradual degradation, and whether larger models are more sensitive to data poisoning.

## Prerequisites

- Python 3.10+ on PATH (verified here with `python3`)
- ~200 MB disk for venv
- CPU only, no GPU required
- No API keys or authentication needed
- Runtime: `run.py` completes in about 1-2 minutes on CPU in the verification environment used for this PR

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/data-poisoning/
```

All subsequent commands assume you are in this directory.

## Step 1: Create virtual environment and install dependencies

```bash
cd submissions/data-poisoning
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Expected output:** Packages install without errors. Key deps: `torch==2.6.0`, `numpy==2.2.4`, `scipy==1.15.2`, `matplotlib==3.10.1`, `pytest==8.3.5`.

## Step 2: Run unit tests

```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** Pytest exits with `31 passed` and exit code 0. Tests cover data generation, label poisoning, MLP training, accuracy evaluation, result aggregation, and sigmoid curve fitting.

## Step 3: Run the experiment

```bash
.venv/bin/python run.py
```

**Expected output:** 81 training runs (9 poison fractions x 3 model widths x 3 seeds) complete in about 1-2 minutes on CPU in the verification environment used for this PR. Output includes:
- Progress updates every 9 runs
- Sigmoid fit parameters (k, x0, threshold, R-squared) per model size
- Key findings: critical thresholds, steepness, larger-model sensitivity
- Files saved to `results/`: `results.json`, `accuracy_vs_poison.png`, `generalization_gap.png`, `train_vs_test.png`

Example findings:
```
Critical thresholds (midpoint between clean and chance):
  Width 32: 43.4% poison
  Width 64: 37.3% poison
  Width 128: 34.9% poison
Larger models: MORE SENSITIVE to poisoning (lower threshold)
```

## Step 4: Validate results

```bash
.venv/bin/python validate.py
```

**Expected output:** `VALIDATION PASSED — all checks OK`. Validates:
- All output files exist (`results.json`, `performance.json`, and 3 PNG plots)
- 81 runs, 27 aggregated points, 3 sigmoid fits
- Clean accuracy > 0.7 for all model sizes
- Accuracy degrades at 50% poison
- Monotonically decreasing accuracy vs. poison fraction
- Sigmoid R-squared > 0.8 for all model sizes
- Deterministic scientific results exclude runtime metadata
- Standard deviations reported
- Runtime under 3 minutes

## Experiment Design

| Parameter | Value |
|-----------|-------|
| Data | Synthetic Gaussian clusters, 500 samples, 10 features, 5 classes |
| Cluster std | 2.0 (moderate overlap for non-trivial classification) |
| Center spread | 2.0x standard normal |
| Poison method | Random label flipping (incorrect class chosen uniformly) |
| Poison fractions | 0%, 1%, 5%, 10%, 15%, 20%, 30%, 40%, 50% |
| Models | 2-layer MLP (ReLU), hidden widths: 32, 64, 128 |
| Training | SGD, lr=0.05, 200 epochs, batch_size=64 |
| Seeds | 3 per config (42, 123, 7), data_seed=42 |
| Train/test split | 70/30 |
| Metrics | Clean test accuracy, train accuracy, generalization gap |
| Analysis | Sigmoid fit to accuracy-vs-poison curve; critical threshold = midpoint of clean and chance |
| Total runs | 81 (9 fractions x 3 widths x 3 seeds) |

## Key Results

1. **Sharp phase transition exists**: Sigmoid steepness k > 5 for larger models (k=8.3 for width 64, k=7.0 for width 128), indicating a sharp rather than gradual accuracy collapse.

2. **Larger models are more sensitive**: Critical thresholds decrease with model size (32: 43.4%, 64: 37.3%, 128: 34.9%). Larger models memorize poisoned labels more readily, degrading faster.

3. **Generalization gap amplifies**: At 50% poison, gen gap increases with width (32: 0.11, 64: 0.24, 128: 0.35), confirming that larger models overfit poisoned data more.

4. **Excellent sigmoid fit**: R-squared > 0.98 for all model sizes, validating that the accuracy-vs-poison relationship follows a sigmoid (logistic) curve.

## Output Files

| File | Description |
|------|-------------|
| `results/results.json` | Deterministic scientific results: config, 81 runs, 27 aggregated points, 3 sigmoid fits, findings |
| `results/performance.json` | Runtime metadata for the latest execution (kept separate from scientific results for reproducibility) |
| `results/accuracy_vs_poison.png` | Test accuracy vs. poison fraction with sigmoid fits and threshold markers |
| `results/generalization_gap.png` | Generalization gap vs. poison fraction per model size |
| `results/train_vs_test.png` | Training vs. test accuracy panel plot (3 model sizes) |

## How to Extend

1. **Different architectures**: Replace `MLP` in `src/model.py` with CNNs, transformers, etc.
2. **Different poisoning strategies**: Modify `poison_labels()` in `src/data.py` for targeted attacks, backdoor triggers, or gradient-based poisoning.
3. **Real datasets**: Replace `generate_gaussian_clusters()` with CIFAR-10, MNIST, etc.
4. **More model sizes**: Add widths to `ExperimentConfig.hidden_widths`.
5. **Defenses**: Add label smoothing, data augmentation, or robust training in `train_model()`.

## Authors

Yun Du, Lina Ji, Claw
