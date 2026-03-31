# Label Noise Tolerance Curves

Sweep label noise (0%--50%) across MLP architectures to measure how network depth and width affect robustness to noisy training labels on synthetic classification data.

## Prerequisites

- Python 3.13 (`python3 --version` reported 3.13.5 in the verified run)
- ~200 MB disk for PyTorch CPU install
- No GPU required; the verified CPU run completed all 168 training runs in 83.9 seconds, so budget about 1-2 minutes depending on machine speed

## Step 0: (Recommended) Start from a clean state

```bash
cd submissions/label-noise
rm -rf .venv results
```

**Expected output:** Command exits with code 0. This ensures a fresh-agent reproduction with no cached artifacts.

## Step 1: Create virtual environment and install dependencies

```bash
cd submissions/label-noise
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

**Expected output:** Successfully installed torch==2.6.0 numpy==2.2.4 scipy==1.15.2 matplotlib==3.10.1 pytest==8.3.5 (plus transitive deps).

## Step 2: Run unit tests

```bash
cd submissions/label-noise
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** Pytest exits with `24 passed` and exit code 0. Tests cover data generation, label noise injection, model construction, training convergence, and evaluation correctness.

## Step 3: Run the full experiment

```bash
cd submissions/label-noise
.venv/bin/python run.py
```

**Expected output:**
- Phase 1: Architecture sweep — 63 runs (7 noise levels x 3 architectures x 3 seeds)
- Phase 2: Width sweep — 105 runs (7 noise levels x 5 widths x 3 seeds)
- Total: 168 training runs in about 1-2 minutes on CPU (verified run: 83.9 seconds)
- Generates: `results/raw_results.json`, `results/summary.json`, `results/arch_sweep.png`, `results/width_sweep.png`
- Prints key findings comparing noise robustness across architectures

## Step 4: Validate results

```bash
cd submissions/label-noise
.venv/bin/python validate.py
```

**Expected output:** `RESULT: PASS` — validates file existence, strict run completeness (exactly 168 runs with no duplicates/missing configs), value ranges, and scientific sanity (noise hurts accuracy, trained models beat chance).

Optional (if results are written elsewhere):

```bash
cd submissions/label-noise
.venv/bin/python validate.py --results-dir /absolute/path/to/results
```

## What This Measures

| Variable | Values |
|----------|--------|
| Label noise fraction | 0%, 5%, 10%, 20%, 30%, 40%, 50% |
| Architecture sweep | shallow-wide (1 layer, h=200), medium (2 layers, h=70), deep-narrow (4 layers, h=35) |
| Width sweep (depth=2) | h=16, 32, 64, 128, 256 |
| Seeds per config | 3 (seeds 42, 43, 44) |
| Dataset | 500 samples, 10 features, 5 Gaussian clusters, 70/30 train/test split |
| Training | 100 epochs, SGD, lr=0.01, batch_size=64, CrossEntropyLoss |
| Metrics | Test accuracy, train accuracy, generalization gap (train - test), all with mean +/- std |

## Key Findings

1. **Deep networks are fragile under noise.** The deep-narrow architecture (4 layers, h=35) starts weak at 0% noise (test acc ~0.54) and collapses to ~0.24 at 50% noise — a 0.31 accuracy drop.
2. **Shallow-wide and medium architectures are robust.** Both maintain >0.85 test accuracy even at 50% noise, with drops of only 0.06--0.09.
3. **Width substantially improves noise tolerance.** In the width sweep (depth=2), h=128 performs best with a 0.042 drop from 0% to 50% noise, h=256 remains strong with a 0.064 drop, and narrow networks (h=16) lose ~0.29.
4. **Noise creates negative generalization gaps.** At high noise, train accuracy tracks the noisy labels (low), but test accuracy on clean labels remains high — producing large negative gaps (train < test) for robust architectures.

## Output Files

| File | Description |
|------|-------------|
| `results/raw_results.json` | Per-run metrics: arch, depth, width, n_params, noise_frac, seed, train_acc, test_acc, gen_gap, wall_seconds (168 entries) |
| `results/summary.json` | Aggregated mean +/- std across seeds, plus auto-derived findings |
| `results/arch_sweep.png` | 3-panel plot: test accuracy, train accuracy, generalization gap vs noise for each architecture |
| `results/width_sweep.png` | 2-panel plot: test accuracy vs noise by width, accuracy drop bar chart |

## How to Extend

1. **Add architectures:** Edit `ARCH_CONFIGS` in `src/models.py` — add a `(depth, width, description)` tuple.
2. **Change noise levels:** Edit `NOISE_FRACS` in `src/experiment.py`.
3. **Try different noise types:** Modify `inject_label_noise()` in `src/data.py` for asymmetric or instance-dependent noise.
4. **Switch datasets:** Replace `build_datasets()` in `src/data.py` with real data loaders (e.g., CIFAR-10).
5. **Add regularization:** Compare noise robustness with/without dropout, weight decay, or mixup in `src/train.py`.
6. **Scale up:** Increase `N_SAMPLES`, `N_EPOCHS`, or `N_FEATURES` in `src/experiment.py`.
