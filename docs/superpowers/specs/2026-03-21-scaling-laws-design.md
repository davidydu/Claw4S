# Scaling Laws Verification — Design Spec (v2)

## Overview

**Submission title:** Scaling Laws Under the Microscope: When Power Laws Predict and When They Don't

**Scientific question:** Do neural scaling laws (Kaplan 2020, Chinchilla 2022) accurately predict the performance of independently-trained public model suites? Where and why do they break down?

**Thesis:** Loss scaling laws are robust and transferable across model families; task-specific scaling laws are not. We provide a rigorous statistical framework that quantifies this gap using published data from Cerebras-GPT and Pythia.

**Approach:** Pure statistical analysis of published training losses and benchmark scores — no model inference, no heavy dependencies, no downloads.

---

## Data Sources

### Primary: Cerebras-GPT (Dey et al., 2023)

- **Paper:** arXiv 2304.03208
- **Models:** 7 sizes — 111M, 256M, 590M, 1.3B, 2.7B, 6.7B, 13B
- **Training:** All trained on The Pile with Chinchilla-optimal ratio (D ≈ 20N). Training tokens: 2.2B, 5.1B, 11.8B, 26.3B, 53.0B, 133B, 257B respectively.
- **Why Cerebras-GPT:** Only public model family with BOTH exact published training losses AND benchmark scores for all sizes. Chinchilla-optimal training means D varies with N, allowing us to test the full two-term Chinchilla formulation.
- **Data points (exact values from HuggingFace model cards):**

| Model | Params | Train Tokens | Pile Test Loss | LAMBADA | HellaSwag | PIQA | WinoGrande | ARC-E | ARC-C | OBQA |
|-------|--------|-------------|----------------|---------|-----------|------|------------|-------|-------|------|
| 111M | 111M | 2.2B | 2.566 | 0.194 | 0.268 | 0.594 | 0.488 | 0.380 | 0.166 | 0.118 |
| 256M | 256M | 5.1B | 2.299 | 0.293 | 0.274 | 0.613 | 0.511 | 0.410 | 0.170 | 0.158 |
| 590M | 590M | 11.8B | 2.184 | 0.366 | 0.291 | 0.627 | 0.498 | 0.464 | 0.190 | 0.158 |
| 1.3B | 1.3B | 26.3B | 1.996 | 0.462 | 0.325 | 0.664 | 0.521 | 0.508 | 0.224 | 0.166 |
| 2.7B | 2.7B | 53.0B | 1.834 | 0.567 | 0.386 | 0.701 | 0.559 | 0.571 | 0.246 | 0.206 |
| 6.7B | 6.7B | 133B | 1.704 | 0.636 | 0.447 | 0.739 | 0.602 | 0.643 | 0.282 | 0.238 |
| 13B | 13B | 257B | 1.575 | 0.696 | 0.513 | 0.766 | 0.646 | 0.714 | 0.367 | 0.286 |

Source: HuggingFace model cards at `huggingface.co/cerebras/Cerebras-GPT-{size}`. All benchmarks are 0-shot.

### Secondary: Pythia (Biderman et al., 2023)

- **Paper:** arXiv 2304.01373
- **Models:** 8 sizes — 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B
- **Training:** All trained on The Pile, same data, same order, ~300B tokens each (fixed D).
- **Purpose:** Cross-family validation for task scaling. Training losses are NOT available as exact published values (only in wandb figures), so Pythia is used for benchmark cross-validation only.
- **Data points:** Downstream task scores from EleutherAI GitHub repo JSON files at `evals/pythia-v1/pythia-{size}/zero-shot/{size}_step143000.json`. Available benchmarks: LAMBADA (acc), WinoGrande (acc), PIQA (acc), ARC-Easy (acc), ARC-Challenge (acc_norm). HellaSwag is NOT in the official Pythia eval suite.
- **Architecture (from paper Table 1 + HuggingFace configs):**

| Model | Total Params | Non-Emb Params | vocab_size | hidden | layers |
|-------|-------------|----------------|------------|--------|--------|
| 70M | ~70M | 18,915,328 | 50,304 | 512 | 6 |
| 160M | ~160M | 85,056,000 | 50,304 | 768 | 12 |
| 410M | ~410M | 302,311,424 | 50,304 | 1,024 | 24 |
| 1B | ~1B | 805,736,448 | 50,304 | 2,048 | 16 |
| 1.4B | ~1.4B | 1,208,602,624 | 50,304 | 2,048 | 24 |
| 2.8B | ~2.8B | 2,517,652,480 | 50,304 | 2,560 | 32 |
| 6.9B | ~6.9B | 6,444,163,072 | 50,432 | 4,096 | 32 |
| 12B | ~12B | 11,327,027,200 | 50,688 | 5,120 | 36 |

Note: Pythia benchmark values must be extracted from the GitHub repo JSON files during implementation and hardcoded in `src/data.py` with exact citations.

### Data Embedding Strategy

All data is hardcoded in `src/data.py` as Python dicts with inline citations. No network calls, no API keys, no downloads. This guarantees determinism and eliminates the most common executability failures.

---

## Analysis Pipeline

### Phase 1: Loss Scaling Verification (Cerebras-GPT)

**Goal:** Confirm that training loss scales predictably with model size, and compare three proposed functional forms.

**Scaling law formulations:**

1. **Kaplan power-law:** L(N) = a·N^(-α) + L∞
   - N = total parameters
   - Three free parameters: a, α, L∞
   - The simplest formulation

2. **Chinchilla two-term:** L(N, D) = a·N^(-α) + b·D^(-β) + L∞
   - For Cerebras-GPT, D varies with N (Chinchilla-optimal: D ≈ 20N), so both terms are active
   - Five free parameters: a, α, b, β, L∞
   - Tests whether separating data-scaling from model-scaling improves fit
   - Note: For Pythia (fixed D ≈ 300B), this reduces to the Kaplan form since b·D^(-β) is constant. This reduction is itself a methodological finding worth reporting.

3. **Power-law with finite-size correction:** L(N) = a·N^(-α)·(1 + c·N^(-γ)) + L∞
   - Adds a correction term that matters at small scale but vanishes at large scale
   - Five free parameters: a, α, c, γ, L∞
   - Motivated by Pearce et al. (2024) reconciliation showing finite-sample artifacts in Kaplan exponents

**Fitting methodology:**
- Nonlinear least squares (scipy.optimize.curve_fit) with multiple random restarts (10+) to avoid local minima
- **Parametric bootstrap** (B=1000): simulate data from fitted model + estimated residual distribution, refit to each simulated dataset. More appropriate than nonparametric bootstrap for n=7 data points.
- Fixed random seed for reproducibility
- Explicit handling of convergence failures: catch RuntimeError from curve_fit, log failed iterations, report fraction of bootstrap samples that converged. Require ≥ 90% convergence for CIs to be considered reliable.
- All fitting performed in **log-log space** (log(L) vs log(N)), which is standard for scaling laws and stabilizes the fit

**Model selection:**
- AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) for each formulation
- **Adjusted R²** for goodness-of-fit (accounts for number of parameters relative to n=7)
- Leave-one-out cross-validation error

### Phase 2: Task Scaling Analysis (Cerebras-GPT + Pythia)

**Goal:** Demonstrate that downstream task accuracy does NOT follow clean scaling laws, in contrast to loss.

**For each task (LAMBADA, HellaSwag, PIQA, WinoGrande, ARC-E, ARC-C, OBQA):**

1. Fit bounded power-law: acc(N) = 1 - a·N^(-α) (naturally bounded in [0,1] for positive a, α)
2. Fit logistic/sigmoid: acc(N) = L / (1 + exp(-k·(log(N) - x₀))) (bounded, captures saturation)
3. Fit piecewise linear (breakpoint detection): exhaustive search over all possible breakpoints between consecutive model sizes. Two-segment linear regression, select breakpoint minimizing total RSS. Acknowledge low statistical power with n=7.
4. Report adjusted R², AIC/BIC, residual analysis for each

**Cross-family comparison:** Fit the same formulations to Pythia benchmark data (5 overlapping tasks). Compare fitted exponents and R² values across families.

**Expected finding:** Task scaling shows worse fits (lower adj. R², wider CIs) compared to loss scaling. Some tasks may show nonmonotonic scaling (e.g., WinoGrande) or near-chance behavior at small scale.

### Phase 3: Cross-Metric Correlation

**Goal:** Quantify how well loss improvement predicts task improvement.

- For each consecutive pair of model sizes, compute Δloss and Δaccuracy (6 data points for Cerebras-GPT)
- Pearson and Spearman rank correlation between Δloss and Δaccuracy
- Also compute rank correlation across all model sizes (not just consecutive deltas) as a complementary measure
- Acknowledge low statistical power with 6-7 data points; report p-values and treat as exploratory
- Expected finding: correlation is moderate overall but may be nonlinear

### Phase 4: Extrapolation Risk Analysis

**Goal:** Quantify how far you can extrapolate scaling laws.

- Fit scaling law on the 4 smallest Cerebras-GPT models (111M–1.3B)
- Predict loss/accuracy for the 3 largest models (2.7B–13B)
- Report prediction error (MAPE) and prediction interval width from parametric bootstrap
- Compare: loss extrapolation error (expected: small, <5%) vs. task extrapolation error (expected: large, >20%)
- This directly quantifies the practical risk of using scaling laws for compute planning

### Phase 5: Cross-Family Transfer

**Goal:** Test universality — do scaling law parameters fitted on one model family predict another?

- Fit loss scaling law on Cerebras-GPT
- Predict Pythia training losses — **not directly possible** since Pythia training losses are unavailable as exact values
- Alternative approach: fit task scaling on Cerebras-GPT, predict Pythia benchmark scores (5 overlapping tasks)
- Compare fitted exponents between families: if task exponents are consistent, that supports partial universality even for task scaling
- If exponents differ, characterize how and hypothesize why (training recipe, data order, architecture differences)

---

## Module Structure

```
submissions/scaling-laws/
├── SKILL.md                    # Executable skill
├── run.py                      # Main entry point
├── validate.py                 # Results validator
├── requirements.txt            # Pinned deps (numpy, scipy, matplotlib, pytest)
├── conftest.py                 # Pytest config
├── src/
│   ├── __init__.py
│   ├── data.py                 # Hardcoded published results with citations
│   ├── scaling_models.py       # L(N) formulations: Kaplan, Chinchilla, corrected power-law
│   ├── fitting.py              # Nonlinear least squares + parametric bootstrap CIs
│   ├── model_selection.py      # AIC/BIC computation, adjusted R², leave-one-out CV
│   ├── task_analysis.py        # Task scaling fits + breakpoint detection
│   ├── extrapolation.py        # Train-on-small, predict-large analysis
│   ├── cross_family.py         # Cross-family transfer analysis
│   ├── plots.py                # All matplotlib figures
│   └── report.py               # Markdown report generator
├── tests/
│   ├── __init__.py
│   ├── test_scaling_models.py  # Test formulation math on synthetic data
│   ├── test_fitting.py         # Test fitting recovers known parameters
│   ├── test_model_selection.py # Test AIC/BIC computation
│   ├── test_data.py            # Test data integrity (counts, ranges, bounds)
│   └── test_task_analysis.py   # Test task fitting on synthetic data
├── research_note/
│   └── main.tex
└── results/                    # gitignored
    ├── results.json
    ├── report.md
    └── figures/
        ├── loss_scaling.png
        ├── task_scaling.png
        ├── residuals.png
        ├── model_selection.png
        └── extrapolation.png
```

---

## Key Figures

1. **Loss vs. Parameters (log-log)** — Three fitted curves with bootstrap CI bands overlaid on Cerebras-GPT data points (7 points). Classic scaling law visualization. Should show tight fit.

2. **Task Accuracy vs. Parameters** — 7 downstream tasks, fitted bounded-power-law curves with CIs. Subplots per task. Should show poor/noisy fit for several tasks.

3. **Residual Comparison** — Side-by-side residual plots for loss scaling (small, random) vs. task scaling (large, structured). Visually demonstrates the gap.

4. **Model Selection (AIC/BIC)** — Bar chart comparing the three loss scaling formulations. Shows which functional form is statistically preferred.

5. **Extrapolation Risk** — Prediction intervals for large models when fitting on small models. Loss intervals stay narrow; task intervals blow up.

---

## Dependencies

```
numpy==2.2.4
scipy==1.15.2
matplotlib==3.10.1
pytest==8.3.5
```

No PyTorch, no transformers, no datasets, no network calls. Install time: ~10 seconds.

Note: Version pins match compatible recent stable releases. Will verify against latest stable at implementation time.

---

## Expected Runtime

- pip install: ~10 seconds
- pytest: ~5 seconds
- run.py (all 5 analysis phases): ~30-45 seconds (parametric bootstrap is the bottleneck)
- Total: well under 2 minutes

---

## Validation Criteria (validate.py)

- [ ] `results/results.json` exists and is valid JSON
- [ ] All 3 loss scaling formulations have finite fitted parameters and CIs
- [ ] ≥90% of bootstrap samples converged for each formulation
- [ ] Loss scaling best-fit adjusted R² > 0.90 (in log-log space)
- [ ] At least 2 of 7 tasks show adjusted R² < 0.85 for bounded power-law fit
- [ ] All 5 expected figures exist as PNG files
- [ ] `results/report.md` exists and contains expected sections
- [ ] Fitted Kaplan exponent α is within [0.02, 0.20] (wide sanity check; Kaplan reports ~0.076, exact value depends on methodology)
- [ ] Bootstrap produced ≥ 900 valid samples (out of 1000)

Note: Thresholds (R² > 0.90, R² < 0.85, α range) should be verified with a pilot run during implementation. The values above are informed estimates.

---

## Error Handling

- `curve_fit` convergence failures: catch `RuntimeError`, log the failure, continue with remaining analyses
- All phases are independent: a failure in one phase does not block others
- `validate.py` reports which phases succeeded and which failed
- Bootstrap degenerate fits (nonsensical parameters): filter by parameter bounds before computing CIs

---

## How to Extend

- **Add a model family:** Add a new dict to `src/data.py` following the existing format. Run `run.py` — it auto-detects all families.
- **Add a downstream task:** Add accuracy values to the model dicts in `data.py`. The task analysis auto-discovers all task keys.
- **Add a scaling formulation:** Add a function to `src/scaling_models.py` following the existing signature. Register it in the formulation list.
- **Change bootstrap samples:** Adjust `N_BOOTSTRAP` in `src/fitting.py`.
- **Add training loss data:** If Pythia or other families publish exact training loss values in the future, add them to `data.py` and loss scaling auto-includes them.

---

## Research Note Outline

1. **Introduction:** Scaling laws are foundational to LLM development but their reliability varies. We distinguish loss scaling (robust) from task scaling (unreliable). Key references: Kaplan 2020, Chinchilla 2022, Pearce 2024, "Scaling Laws Are Unreliable" EMNLP 2025.
2. **Data:** Cerebras-GPT (7 sizes, Chinchilla-optimal) and Pythia (8 sizes, fixed data budget). All data publicly published.
3. **Methods:** Three scaling formulations, parametric bootstrap CIs, AIC/BIC model selection, breakpoint detection, extrapolation risk quantification.
4. **Results:** Loss scaling confirms power-law with α ≈ X [95% CI]. Task scaling shows poor fits for Y of Z tasks. Extrapolation risk is N× larger for tasks than losses. Cross-family exponents show [consistency/divergence].
5. **Discussion:** Implications for compute planning, benchmark design, and the "emergent abilities" debate.
6. **Conclusion + How to Extend.**
