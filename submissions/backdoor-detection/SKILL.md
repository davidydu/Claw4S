---
name: backdoor-detection-spectral-signatures
description: Detect backdoor (trojan) attacks in neural networks using spectral signatures of penultimate-layer activations. Trains clean and backdoored MLPs on synthetic Gaussian cluster data, then applies eigenvalue decomposition of the activation covariance matrix to identify poisoned samples (Tran et al. 2018). Sweeps over poison fraction, trigger strength, and model size (36 experiments).
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Backdoor Detection via Spectral Signatures

This skill reproduces and extends the spectral signature method for neural network backdoor detection (Tran et al. 2018). It trains clean and trojaned two-layer MLPs on synthetic data, extracts penultimate-layer activations, and detects poisoned samples via the top eigenvector of the activation covariance matrix.

## Prerequisites

- Requires **Python 3.10+** (CPU only, no GPU needed).
- Expected runtime: **1-2 minutes**.
- All commands must be run from the **submission directory** (`submissions/backdoor-detection/`).
- No internet access required (all data is synthetically generated).
- No API keys or authentication needed.

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

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `34 passed` and exit code 0.

## Step 3: Run the Experiment Sweep

Execute the full 36-experiment parameter sweep:

```bash
.venv/bin/python run.py
```

This will:
1. Generate synthetic Gaussian cluster data (500 samples, 10 features, 5 classes)
2. For each of 36 configurations (4 poison fractions x 3 trigger strengths x 3 model sizes):
   - Inject backdoor trigger (set features 0-2 to fixed values, relabel to target class)
   - Train clean and backdoored 2-layer MLPs
   - Extract penultimate-layer activations from the backdoored model
   - Compute spectral scores via top eigenvector of activation covariance
   - Measure detection AUC (ROC AUC for identifying poisoned samples)
3. Generate report and figures

Expected output: Each experiment prints its config and detection AUC. The script prints `[4/4] Generating figures...` and exits with code 0. Files created in `results/`:
- `results.json` — full experiment data
- `report.md` — markdown summary with tables
- `fig_auc_heatmap.png` — AUC heatmap (poison fraction vs trigger strength)
- `fig_auc_by_model_size.png` — AUC vs poison fraction by model size
- `fig_eigenvalue_ratio.png` — spectral gap vs poison fraction

For reproducibility, `results.json` excludes wall-clock timing fields.

## Step 4: Validate Results

```bash
.venv/bin/python validate.py
```

Expected output: `VALIDATION PASSED: All checks OK` with exit code 0. The validator checks:
- All 36 experiments completed
- AUC values in [0, 1]
- Clean model accuracy > 50% (better than random)
- All output files exist and are non-empty
- Thesis check: experiments with strong triggers (strength=10.0) and poison fraction >= 10% achieve AUC >= 0.9

## Key Parameters

| Parameter | Values | Purpose |
|-----------|--------|---------|
| Poison fraction | 5%, 10%, 20%, 30% | Fraction of training data with trigger |
| Trigger strength | 3.0, 5.0, 10.0 | Magnitude of trigger pattern |
| Hidden dim | 64, 128, 256 | Model capacity |
| Samples | 500 | Total training samples |
| Features | 10 | Input dimensionality |
| Classes | 5 | Number of target classes |
| Seed | 42 | Reproducibility |

## Expected Findings

- **Joint phase transition in detectability**: trigger strength is the dominant factor, but poison fraction matters. With `strength=10.0` and poison fraction >= 10%, all 9 such experiments achieve AUC >= 0.9 across model sizes. At 5% poison, even `strength=10.0` stays near-random in the reference run (AUC 0.166-0.281).
- Weaker triggers (`strength=3.0` or `5.0`) evade spectral detection in all 24 experiments (all AUC < 0.5).
- Poison fraction has a secondary effect: higher fractions increase the chance that a strong trigger becomes spectrally separable, but they do not rescue weak-trigger detection.
- Model size (hidden dim) has modest effect on detectability.
- The spectral gap shows an overall increase from low to high poison fractions, but not strict monotonic growth at every step.
- 9/36 experiments achieve AUC >= 0.9; they are exactly the `strength=10.0`, poison fraction >= 10% cases.

## How to Extend

1. **Different architectures**: Replace the MLP in `src/model.py` with a CNN or transformer; the spectral analysis in `src/spectral.py` works on any layer's activations.
2. **Real datasets**: Replace `generate_clean_data()` in `src/data.py` with a real dataset loader (e.g., CIFAR-10). The trigger injection logic generalizes to image patches.
3. **Other detection methods**: Add alternative detectors (e.g., activation clustering, Neural Cleanse) alongside spectral signatures in `src/spectral.py`.
4. **Adaptive attacks**: Modify `inject_backdoor()` to use learned or stealthy triggers that evade spectral detection.
5. **Statistical analysis**: Add bootstrap confidence intervals or multiple random seeds by modifying `run_sweep()` in `src/experiment.py`.
