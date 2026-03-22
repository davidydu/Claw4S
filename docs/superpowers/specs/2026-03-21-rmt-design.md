# Design Spec: Random Matrix Theory for Neural Network Weights

## Motivation

Random Matrix Theory (RMT) provides a null model for understanding what structure
trained neural networks learn. The Marchenko-Pastur (MP) distribution describes
the eigenvalue spectrum of W^T W when W is a random matrix with i.i.d. entries.
After training, layers that learn meaningful structure deviate from this null
distribution. The degree of deviation quantifies how much information the layer
has captured.

This submission trains tiny MLPs on modular arithmetic (a well-studied task that
exhibits grokking) and simple regression, then analyzes weight matrix spectra
using RMT tools. The key hypothesis: **layers that deviate more from
Marchenko-Pastur have learned more task-relevant structure**.

## Background

### Marchenko-Pastur Distribution

For an M x N random matrix W with i.i.d. entries of mean 0 and variance sigma^2,
the eigenvalue distribution of (1/M) W^T W converges to the MP law:

    rho(lambda) = (1 / (2*pi*sigma^2*gamma)) * sqrt((lambda_+ - lambda)(lambda - lambda_-)) / lambda

where:
- gamma = N/M (aspect ratio)
- lambda_+/- = sigma^2 * (1 +/- sqrt(gamma))^2 (bulk edges)

Eigenvalues outside [lambda_-, lambda_+] represent signal (structure learned
beyond random).

### Prior Work

- Martin & Mahoney (2019, 2021): Empirical spectral density of DNN weight
  matrices displays self-regularization signatures. Identified 5+1 phases of
  training via heavy-tailed universality classes.
- Pennington & Worah (2017): RMT for loss surface geometry.
- Recent (2022-2024): RMT applied to locate learned information in transformers
  by identifying eigenvectors that deviate from Porter-Thomas distribution.

## Scientific Questions

1. Do trained MLP weight matrices deviate from MP predictions?
2. Is the deviation proportional to layer importance (early vs late layers)?
3. How does network width affect spectral properties?
4. Do untrained (random init) weights match MP as expected?

## Experimental Design

### Tasks
1. **Modular Addition (mod 97):** Input: two integers (a, b) in [0, 96],
   one-hot encoded. Output: (a + b) mod 97. Classification (97 classes).
2. **Polynomial Regression:** Input: x in [-1, 1]. Output: polynomial
   f(x) = 0.5*x^3 - 0.3*x^2 + 0.7*x - 0.1. Regression (1 output).

### Models
Tiny MLPs with varying hidden dimensions: 32, 64, 128, 256.
Architecture: Input -> Linear -> ReLU -> Linear -> ReLU -> Linear -> Output.
Three layers of weights to analyze per model.

### Training
- Optimizer: Adam, lr=1e-3
- Modular addition: 2000 epochs, cross-entropy loss
- Regression: 1000 epochs, MSE loss
- Seed: 42 (deterministic)
- CPU only, no CUDA

### Analysis Pipeline
For each weight matrix W (shape M x N):
1. Compute correlation matrix C = (1/M) * W^T W
2. Compute eigenvalues of C
3. Fit MP distribution using empirical sigma^2 and gamma = N/M
4. Compute KS statistic between empirical CDF and theoretical MP CDF
5. Count fraction of eigenvalues outside MP bulk [lambda_-, lambda_+]
6. Compare trained vs untrained (random init) spectra

### Metrics
- **KS Statistic:** Kolmogorov-Smirnov distance between empirical and MP CDFs.
  Higher = more deviation from random.
- **Outlier Fraction:** Fraction of eigenvalues outside MP bulk edges.
  Higher = more learned structure.
- **Spectral Norm Ratio:** Largest eigenvalue / lambda_+. Ratio > 1 indicates
  signal spikes beyond random bulk.
- **KL Divergence (binned):** Approximate KL divergence between binned
  empirical and MP PDFs.

### Expected Results
- Untrained networks: KS statistic < 0.1, outlier fraction near 0
- Trained networks: KS statistic > 0.2 for later layers
- Wider networks (256 hidden): Better MP fit for untrained (finite-size effects
  decrease)
- Modular arithmetic: Stronger deviation than regression (more structured task)

## Output Artifacts
- `results/results.json`: All metrics per (model, layer, trained/untrained)
- `results/eigenvalue_spectra.png`: Eigenvalue histograms vs MP overlay
- `results/ks_summary.png`: KS statistics across layers and widths
- `results/report.md`: Human-readable summary

## Module Design

| Module | Responsibility |
|--------|---------------|
| `src/data.py` | Generate modular arithmetic and regression datasets |
| `src/model.py` | Define MLP architecture |
| `src/train.py` | Training loop with deterministic seeding |
| `src/rmt_analysis.py` | Eigenvalue computation, MP fitting, KS/KL metrics |
| `src/plots.py` | Matplotlib visualizations |
| `src/report.py` | Generate markdown report from results |
| `run.py` | Orchestrate full pipeline |
| `validate.py` | Validate results completeness and correctness |

## Constraints
- Runtime <= 3 minutes on CPU
- No CUDA/GPU dependencies
- All deps pinned with ==
- Seed=42 everywhere for reproducibility
- No network access required (synthetic data only)
