# Double Descent in Practice — Design Spec

## 1. Scientific Goal

Systematically reproduce the **double descent phenomenon** (Nakkiran et al. 2019, Belkin et al. 2019) using small MLPs on synthetic data, running entirely on CPU in under 3 minutes. We demonstrate:

1. **Model-wise double descent**: Test error first decreases, then increases at the interpolation threshold (#params ~ #samples), then decreases again as width grows further.
2. **Epoch-wise double descent**: At the interpolation threshold, test error shows a U-shape followed by a second descent over training epochs.
3. **Label noise amplification**: More label noise makes the double descent peak more pronounced.

## 2. Key References

- **Nakkiran et al. 2019** "Deep Double Descent": Model-wise and epoch-wise double descent across architectures. Label noise amplifies the peak. The interpolation threshold is where #effective_params ~ #samples.
- **Belkin et al. 2019** "Reconciling modern ML with bias-variance trade-off": The double descent curve subsumes the classical U-shaped curve. Interpolation threshold is N=n for regression.
- **Advani & Saxe 2017**: Two-layer networks with random first-layer weights exhibit double descent analytically. Overtraining is worst when #params ~ #samples.

## 3. Experimental Design

### 3.1 Dataset: Synthetic Noisy Regression

- **X**: n=150 samples, d=20 features, drawn from N(0,1)
- **True function**: y = X @ w_true + epsilon, where w_true ~ N(0,1), epsilon ~ N(0, sigma)
- **Train/test split**: 100 train, 50 test (fixed seed=42)
- **Label noise levels**: sigma in {0.1, 0.5, 1.0} (low, medium, high noise)

Why 100 train samples: The interpolation threshold for a 2-layer MLP with hidden width h and d=20 input features is #params = h*(d+1) + (h+1) = h*21 + h + 1 = 22h + 1. So threshold is at h ~ 100/22 ~ 5. But we want to sweep h from underfitting to heavily overparameterized, so we use h from 2 to 500. The threshold region is around h=5-10 with this parameterization.

**Revised**: To get a clearer double descent with the threshold at a visible width, we use:
- n_train = 200, n_test = 100, d = 10
- MLP: input(10) -> hidden(h) -> output(1), so #params = h*11 + h + 1 = 12h + 1
- Interpolation threshold: h ~ 200/12 ~ 17
- Sweep h in: [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30, 40, 60, 80, 100, 150, 200, 300, 500]
- This gives ~20 model sizes spanning under-parameterized through interpolation to over-parameterized

### 3.2 Model: 2-Layer MLP

```
Input(d=10) -> Linear(d, h) -> ReLU -> Linear(h, 1)
```

- Parameters: h*(d+1) + (h+1) = 12h + 1
- Trained with MSE loss, SGD (lr=0.01, no momentum), for fixed epochs
- No regularization (no weight decay, no dropout) — this is critical for double descent

### 3.3 Experiment 1: Model-Wise Double Descent

- Fix training epochs = 2000 (enough to converge/overfit)
- Sweep hidden width h across 20 values
- For each h: train model, record final train MSE and test MSE
- Repeat for 3 noise levels
- **Expected**: Test MSE peaks near h=17 (interpolation threshold), especially with high noise

### 3.4 Experiment 2: Epoch-Wise Double Descent

- Fix h at interpolation threshold (h ~ 17)
- Train for 5000 epochs, recording test MSE every 50 epochs
- **Expected**: Test MSE decreases, then increases (overfitting), then decreases again

### 3.5 Experiment 3: Phase Diagram

- Sweep both width h (subset of values) and epochs
- Create 2D heatmap of test MSE as function of (width, epoch)
- **Expected**: The peak region forms a curve in the (width, epoch) plane

## 4. Outputs

| File | Description |
|------|-------------|
| `results/results.json` | All experimental data: per-width, per-epoch metrics |
| `results/model_wise_double_descent.png` | Test MSE vs model width for each noise level |
| `results/epoch_wise_double_descent.png` | Test MSE vs epoch at interpolation threshold |
| `results/phase_diagram.png` | 2D heatmap of test MSE (width x epochs) |
| `results/noise_comparison.png` | Overlay of double descent curves at different noise levels |
| `results/report.md` | Summary of findings |

## 5. Success Criteria

1. Model-wise curve shows clear non-monotonic behavior (decrease-increase-decrease) in test error
2. The peak aligns with the interpolation threshold (where #params ~ #train_samples)
3. Higher label noise produces more pronounced double descent peak
4. Epoch-wise curve shows secondary descent after initial overfitting
5. Total runtime < 3 minutes on CPU

## 6. Risk Mitigation

- **Double descent not visible**: Increase label noise (the phenomenon requires noise). Use SGD without regularization. Train long enough to interpolate.
- **Too slow**: Reduce number of width values or max epochs. Use smaller n_train.
- **Numerical instability**: Normalize inputs. Use moderate learning rate.
