# Design Spec: Memorization Capacity Scaling

## Motivation

Zhang et al. (2017) showed that deep neural networks can perfectly memorize random labels -- a surprising result that challenged conventional understanding of generalization. Their key insight: networks with sufficient parameters can interpolate any training set, regardless of label structure. The *interpolation threshold* -- the point where #params ~ #samples -- determines when memorization becomes possible.

This submission systematically measures the interpolation threshold for MLPs on synthetic data and characterizes whether the transition from under-memorization to full memorization is sharp (phase-transition-like) or gradual.

## Research Questions

1. At what parameter count does a 2-layer MLP achieve 100% training accuracy on random labels?
2. Is the transition from partial to full memorization sharp or gradual?
3. How does the interpolation threshold differ between random labels and structured (true) labels?

## Experimental Design

### Dataset
- Synthetic: X ~ N(0, 1), shape (n, d) where n=200 samples, d=20 features
- Random labels: y ~ Uniform({0, ..., 9}), 10 classes
- True labels: y = cluster assignment from k-means on X (10 clusters)
- Fixed seed=42 for reproducibility

### Model
- 2-layer MLP: Linear(d, h) -> ReLU -> Linear(h, 10)
- Hidden widths h: [5, 10, 20, 40, 80, 160, 320, 640]
- Parameter count formula: d*h + h + h*10 + 10 = h*(d + 10) + d + 10 = h*30 + 30 (for d=20)
  - h=5: 180 params
  - h=10: 330 params
  - h=20: 630 params
  - h=40: 1230 params
  - h=80: 2430 params
  - h=160: 4830 params
  - h=320: 9630 params
  - h=640: 19230 params
- Interpolation threshold prediction: n=200 samples * 10 classes needs ~200 effective params minimum, so threshold should be near h=10-40 range

### Training
- Optimizer: Adam, lr=0.001
- Loss: CrossEntropyLoss
- Max epochs: 5000
- Convergence criterion: training loss < 1e-4 OR 100% training accuracy for 10 consecutive epochs
- Batch size: full batch (n=200 fits in memory)
- seed=42, torch manual_seed + numpy seed

### Metrics
- Training accuracy (memorization capacity)
- Test accuracy on held-out data with same label structure (50 samples)
- Convergence epoch (how fast memorization happens)
- Parameter count at interpolation threshold

### Analysis
- Plot train_acc vs log(#params) for both label types
- Fit sigmoid to train_acc vs log(#params): acc = 1 / (1 + exp(-k*(log(p) - log(p_threshold))))
- k measures sharpness: large k = sharp transition, small k = gradual
- Report p_threshold (interpolation threshold) and k (sharpness) for both label types
- Compare thresholds: random labels should require more params than true labels

## Expected Findings

1. Random labels: full memorization when #params >> n (likely h >= 40-80)
2. True labels: full memorization with fewer params (structured data is easier)
3. Transition should be moderately sharp (sigmoid k ~ 2-5 in log-param space)
4. Test accuracy should remain near chance (10%) for random labels regardless of model size

## Scope & Constraints

- CPU-only PyTorch, no CUDA
- Total runtime < 3 minutes (16 training runs, each < 10s)
- No external datasets (synthetic data generation)
- No pretrained models
- Fully deterministic with fixed seeds
