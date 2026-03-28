---
name: feature-attribution-consistency
description: Measure pairwise agreement (Spearman rank correlation) between three gradient-based attribution methods (vanilla gradient, gradient x input, integrated gradients) across MLP depths on synthetic classification data.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Feature Attribution Consistency

This skill trains small MLPs of varying depth on synthetic Gaussian cluster data, computes three gradient-based feature attribution methods on test samples, and measures pairwise Spearman rank correlation to quantify attribution agreement. The experiment sweeps 3 depths x 3 method pairs x 100 samples x 3 seeds.

## Prerequisites

- Requires **Python 3.10+**. CPU only, no GPU required.
- No internet access needed (fully synthetic data).
- Expected runtime: **1-3 minutes**.
- All commands must be run from the **submission directory** (`submissions/feature-attribution/`).

## Step 1: Environment Setup

Create a virtual environment and install pinned dependencies:

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

Verify all modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: All tests pass (`X passed` with exit code 0). Tests cover data generation, model training, attribution computation, and agreement metrics.

## Step 3: Run the Experiment

Execute the full attribution consistency analysis:

```bash
.venv/bin/python run.py
```

Expected output includes per-depth accuracy and Spearman correlation tables, ending with:
```
Results saved to results/results.json
Report saved to results/report.md

Experiment complete.
Overall mean Spearman rho: <value>
Substantial disagreement: <True/False>
```

This will:
1. Generate synthetic Gaussian cluster data (500 samples, 10 features, 5 classes)
2. Train MLPs with 1, 2, and 4 hidden layers (width=64) for each of 3 seeds
3. Compute vanilla gradient, gradient x input, and integrated gradients on 100 test samples with respect to each model's predicted class logit
4. Measure pairwise Spearman rank correlation between all method pairs
5. Aggregate statistics across samples and seeds
6. Save results to `results/results.json` and `results/report.md`

## Step 4: Validate Results

```bash
.venv/bin/python validate.py
```

Expected output ends with: `VALIDATION PASSED: All checks OK` and exit code 0.

Validates:
- All 3 depths and 3 seeds are present
- All 3 method pairs have correlation data
- Correlations are in valid range [-1, 1]
- Model accuracies are above 50%
- Report file exists

## Expected Results

- **Model accuracy**: >90% for all depths (Gaussian clusters are well-separated)
- **Attribution agreement**: Spearman rho varies by method pair
  - Gradient x Input vs Integrated Gradients: highest agreement (typically ~0.93-0.97)
  - Vanilla Gradient vs others: moderate agreement (typically ~0.68-0.78)
- **Depth effect**: method-pair-dependent in this configuration; vanilla-gradient agreement tends to rise modestly with depth while GI-IG remains consistently high

## How to Extend

1. **More depths**: Edit `depths` list in `src/experiment.py:run_experiment()`
2. **Different data**: Replace `make_gaussian_clusters()` in `src/data.py` with any (X, y) generator
3. **New attribution methods**: Add to `src/attributions.py:METHODS` dict and update `METHOD_PAIRS`
4. **Real datasets**: Swap the data module; all downstream code works with any (n, d) tensor input
5. **Different models**: Replace `MLP` class in `src/models.py`; attributions only need `model.forward()`
