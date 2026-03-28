# Skill: Membership Inference Under Differential Privacy

Reproduce an experiment showing that DP-SGD empirically reduces membership inference attack success in this controlled setting. Train 2-layer MLPs on synthetic Gaussian cluster data with four privacy levels (non-private, weak/moderate/strong DP), then run shadow-model membership inference attacks (Shokri et al. 2017) against each. Measure attack AUC, model utility, and the privacy-utility-leakage triad.

**Key finding:** On the verified March 28, 2026 runs, DP-SGD with strong privacy (sigma=5.0, epsilon~3.4) reduces membership inference AUC from 0.664 to 0.518 (near random guessing at 0.5), a reduction of 0.146.

## Prerequisites

- Python 3.11+ with `pip`
- ~500 MB disk (PyTorch CPU)
- CPU only; no GPU required
- No API keys or authentication needed
- Runtime: about 35 seconds wall-clock on a modern laptop CPU; budget up to 1 minute on slower machines

## Step 1: Set Up Virtual Environment

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

**Expected output:** Successfully installed torch-2.6.0, numpy-2.2.4, scipy-1.15.2, matplotlib-3.10.1, pytest-8.3.5 (plus dependencies).

## Step 2: Run Unit Tests

```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** All 28 tests pass. Key test groups:
- `test_data.py` (6 tests) — synthetic data generation, member/non-member split, reproducibility, no overlap
- `test_model.py` (3 tests) — MLP forward pass, shape checks, weight reproducibility
- `test_dp_sgd.py` (8 tests) — per-sample gradients, gradient clipping, noise injection, epsilon accounting
- `test_train.py` (3 tests) — standard + DP training, evaluation
- `test_attack.py` (6 tests) — attack features, classifier training, attack metrics
- `test_runtime.py` (2 tests) — script working-directory guard behavior

## Step 3: Run Full Experiment

```bash
.venv/bin/python run.py
```

This runs the complete experiment (about 35 seconds wall-clock on the verified CPU-only runs):
1. For each of 4 privacy levels x 3 seeds = 12 configurations:
   - Generate 500-sample synthetic classification data (10 features, 5 classes, Gaussian clusters)
   - Train target model (2-layer MLP, hidden=128, 80 epochs)
   - Train 3 shadow models with same DP config on fresh data
   - Extract attack features (softmax, confidence, entropy, loss, correctness)
   - Train attack classifier on shadow model features
   - Run membership inference attack against target model
2. Aggregate results and generate plots

**Expected output:**
```
[1/12] non-private (sigma=0.0), seed=42
  epsilon=inf, test_acc=0.768, attack_auc=0.687
...
[12/12] strong-dp (sigma=5.0), seed=456
  epsilon=3.38, test_acc=0.596, attack_auc=0.516

Results saved to results/results.json
Generated 3 plots in results/

========================================================================
MEMBERSHIP INFERENCE UNDER DIFFERENTIAL PRIVACY — RESULTS
========================================================================
Privacy Level     sigma    epsilon   Test Acc   Attack AUC   Attack Acc
non-private         0.0        inf 0.792+/-0.116 0.664+/-0.060 0.613+/-0.058
weak-dp             0.5       53.5 0.849+/-0.085 0.532+/-0.019 0.520+/-0.012
moderate-dp         2.0        9.4 0.805+/-0.091 0.541+/-0.010 0.529+/-0.009
strong-dp           5.0        3.4 0.709+/-0.118 0.518+/-0.004 0.521+/-0.017
========================================================================
```

**Generated files:**
- `results/results.json` — all per-trial and aggregated metrics
- `results/summary.txt` — human-readable summary table
- `results/attack_auc_vs_privacy.png` — bar chart of attack AUC per privacy level
- `results/privacy_utility_leakage.png` — three-panel privacy-utility-leakage triad
- `results/generalization_gap_vs_attack.png` — overfitting correlates with leakage

## Step 4: Validate Results

```bash
.venv/bin/python validate.py
```

**Expected output:**
```
Privacy levels: 4
Seeds: 3
Total runs: 12 (expected 12)
Non-private attack AUC:  0.664
Strong-DP attack AUC:    0.518
AUC reduction:           0.146
Non-private test accuracy: 0.792
Plot exists: results/attack_auc_vs_privacy.png
Plot exists: results/privacy_utility_leakage.png
Plot exists: results/generalization_gap_vs_attack.png
Validation PASSED.
```

## Method Details

### DP-SGD (Abadi et al. 2016)
Implemented from scratch -- no Opacus or external DP library:
1. **Per-sample gradients** via `torch.func.vmap` + `torch.func.grad`
2. **Per-sample gradient clipping** to L2 norm bound C=1.0
3. **Gaussian noise** with std = sigma * C added to aggregated gradients
4. **Privacy accounting** using simplified RDP (Renyi Differential Privacy) composition, converted to (epsilon, delta)-DP

### Membership Inference Attack (Shokri et al. 2017)
Shadow model approach with enriched features:
1. Train N=3 shadow models per config, each on fresh data with known member/non-member split
2. Extract rich attack features per sample: softmax vector, max confidence, prediction entropy, cross-entropy loss, correctness indicator
3. Train binary neural network attack classifier on shadow model features
4. Apply attack classifier to target model's outputs to infer membership

### Privacy Levels

| Level | sigma | Approx. epsilon | Observed Attack AUC |
|-------|-------|----------------|-------------------|
| Non-private | 0.0 | inf | 0.664 +/- 0.060 (vulnerable) |
| Weak DP | 0.5 | ~53 | 0.532 +/- 0.019 |
| Moderate DP | 2.0 | ~9 | 0.541 +/- 0.010 |
| Strong DP | 5.0 | ~3 | 0.518 +/- 0.004 (near-random) |

## How to Extend

1. **Different architectures:** Replace `MLP` in `src/model.py` with CNNs/Transformers; update `input_dim`, `hidden_dim`, `num_classes` parameters
2. **Real datasets:** Modify `src/data.py` to load CIFAR-10, MNIST, or tabular datasets; adjust `generate_gaussian_clusters()` or add a new data loader
3. **More attack types:** Add loss-threshold or label-only attacks in `src/attack.py` alongside the shadow model approach
4. **Tighter privacy accounting:** Replace RDP in `compute_epsilon()` with Gaussian DP (GDP) or Privacy Loss Distribution (PLD) accounting for tighter epsilon estimates
5. **More privacy levels:** Add entries to `PRIVACY_LEVELS` list in `src/experiment.py`
6. **Different DP mechanisms:** Modify `dp_sgd_step()` in `src/dp_sgd.py` to test alternative clipping strategies (e.g., adaptive clipping) or noise mechanisms

## Limitations

- Synthetic data may not capture real-world distribution complexity
- Small model (2-layer MLP, 128 hidden units) -- larger models may show different DP-utility trade-offs
- Simplified RDP accounting gives upper-bound epsilon estimates; tighter accounting would yield smaller epsilon values
- Shadow model attack assumes attacker knows the model architecture and training procedure
- 3 seeds provides limited statistical power; production studies should use more seeds
