# Scaling Laws Verification — Design Spec

## Overview

**Submission title:** Scaling Laws Under the Microscope: When Power Laws Predict and When They Don't

**Scientific question:** Do neural scaling laws (Kaplan 2020, Chinchilla 2022) accurately predict the performance of independently-trained public model suites? Where and why do they break down?

**Thesis:** Loss scaling laws are robust and transferable across model families; task-specific scaling laws are not. We provide a rigorous statistical framework that quantifies this gap using published data from Pythia and OLMo.

**Approach:** Pure statistical analysis of published training losses and benchmark scores — no model inference, no heavy dependencies, no downloads.

---

## Data Sources

### Primary: Pythia (Biderman et al., 2023)

- **Models:** 8 sizes — 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B
- **Training:** All trained on The Pile, same data, same order, ~300B tokens each
- **Data points to embed:**
  - Final training loss (cross-entropy on The Pile) for each model size
  - Downstream task scores: LAMBADA (accuracy), HellaSwag (acc_norm), WinoGrande (acc), PIQA (acc), ARC-Easy (acc), ARC-Challenge (acc_norm)
- **Source:** Biderman et al. 2023, Tables/Figures from the paper + EleutherAI evaluation results
- **Why Pythia:** Controlled experiment — same data/order eliminates composition confounders

### Secondary: OLMo (Groeneveld et al., 2024)

- **Models:** Available sizes with published losses and evaluations
- **Purpose:** Cross-family validation — test whether scaling exponents transfer
- **Source:** Groeneveld et al. 2024, published evaluation tables

### Data Embedding Strategy

All data is hardcoded in `src/data.py` as Python dicts with inline citations to exact paper tables/figures. No network calls, no API keys, no downloads. This guarantees determinism and eliminates the most common executability failures.

```python
# Example structure (actual values to be extracted from papers)
PYTHIA = {
    "name": "Pythia",
    "source": "Biderman et al., 2023",
    "training_tokens": 300e9,
    "models": {
        "70M":  {"params": 7e7,  "non_emb_params": ..., "final_loss": ..., "lambada_acc": ..., ...},
        "160M": {"params": 1.6e8, ...},
        ...
    }
}
```

---

## Analysis Pipeline

### Phase 1: Loss Scaling Verification

**Goal:** Confirm that training loss scales predictably with model size, and compare three proposed functional forms.

**Scaling law formulations:**

1. **Kaplan power-law:** L(N) = a·N^(-α) + L∞
   - N = non-embedding parameters (following Kaplan's convention)
   - Three free parameters: a, α, L∞

2. **Chinchilla two-term:** L(N, D) = a·N^(-α) + b·D^(-β) + L∞
   - Since all Pythia models train on ~300B tokens, D is fixed, simplifying to effectively 4 free parameters
   - Key difference: separates data-scaling from model-scaling

3. **Power-law with finite-size correction:** L(N) = a·N^(-α)·(1 + c·N^(-γ)) + L∞
   - Adds a correction term that matters at small scale but vanishes at large scale
   - Motivated by the Pearce/Porian reconciliation showing finite-sample artifacts

**Fitting methodology:**
- Nonlinear least squares (scipy.optimize.curve_fit) with multiple random restarts to avoid local minima
- Bootstrap resampling (B=1000) for 95% confidence intervals on all parameters
- Fixed random seed for reproducibility

**Model selection:**
- AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion) for each formulation
- Adjusted R² for goodness-of-fit
- Leave-one-out cross-validation error

### Phase 2: Task Scaling Analysis

**Goal:** Demonstrate that downstream task accuracy does NOT follow clean scaling laws, in contrast to loss.

**For each task (LAMBADA, HellaSwag, WinoGrande, PIQA, ARC-E, ARC-C):**

1. Fit power-law: acc(N) = a·N^α + c
2. Fit logistic/sigmoid: acc(N) = L / (1 + exp(-k·(log(N) - x₀)))
3. Fit piecewise linear (breakpoint detection): two-segment regression with unknown breakpoint
4. Report R², AIC/BIC, residual analysis for each

**Expected finding:** Task scaling shows much worse fits (lower R², wider CIs, AIC doesn't clearly select) compared to loss scaling. Some tasks may show nonmonotonic or trendless scaling.

### Phase 3: Cross-Metric Correlation

**Goal:** Quantify how well loss improvement predicts task improvement.

- For each consecutive pair of model sizes, compute Δloss and Δaccuracy
- Pearson and Spearman correlation between Δloss and Δaccuracy
- Test whether correlation is stable across the size range or breaks at specific scales
- Expected finding: correlation is moderate overall but breaks down at specific transitions

### Phase 4: Extrapolation Risk Analysis

**Goal:** Quantify how far you can extrapolate scaling laws.

- Fit scaling law on the 4 smallest models (70M–1B)
- Predict loss/accuracy for the 4 largest models (1.4B–12B)
- Report prediction error and prediction interval width
- Compare: loss extrapolation error (expected: small) vs. task extrapolation error (expected: large)
- This directly quantifies the practical risk of using scaling laws for planning

### Phase 5: Cross-Family Transfer

**Goal:** Test universality — do scaling law parameters fitted on one model family predict another?

- Fit scaling law on Pythia data
- Predict OLMo losses using Pythia-fitted parameters
- Report transfer prediction error
- If exponents are similar across families, that supports universality of loss scaling
- If they differ, characterize how and why

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
│   ├── fitting.py              # Nonlinear least squares + bootstrap CIs
│   ├── model_selection.py      # AIC/BIC computation, leave-one-out CV
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
│   ├── test_data.py            # Test data integrity (counts, ranges)
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

1. **Loss vs. Parameters (log-log)** — Three fitted curves with bootstrap CI bands overlaid on Pythia data points. Classic scaling law visualization. Should show tight fit.

2. **Task Accuracy vs. Parameters** — Same format but for 4-6 downstream tasks. Fitted curves with CIs. Should show poor/noisy fit for several tasks.

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

---

## Expected Runtime

- pip install: ~10 seconds
- pytest: ~5 seconds
- run.py (all 5 analysis phases): ~30-45 seconds (bootstrap is the bottleneck)
- Total: well under 2 minutes

---

## Validation Criteria (validate.py)

- [ ] `results/results.json` exists and is valid JSON
- [ ] All 3 loss scaling formulations have finite fitted parameters and CIs
- [ ] Loss scaling best-fit R² > 0.95
- [ ] At least 2 tasks show R² < 0.90 for power-law fit (confirming task scaling unreliability)
- [ ] All 5 expected figures exist as PNG files
- [ ] `results/report.md` exists and contains expected sections
- [ ] Fitted Kaplan exponent α is within [0.03, 0.15] (sanity check against published range)
- [ ] Bootstrap produced the expected number of samples

---

## How to Extend

- **Add a model family:** Add a new dict to `src/data.py` following the existing format. Run `run.py` — it auto-detects all families.
- **Add a downstream task:** Add accuracy values to the model dicts in `data.py`. The task analysis auto-discovers all task keys.
- **Add a scaling formulation:** Add a function to `src/scaling_models.py` following the existing signature. Register it in the formulation list.
- **Change bootstrap samples:** Adjust `N_BOOTSTRAP` in `src/fitting.py`.

---

## Research Note Outline

1. **Introduction:** Scaling laws are foundational to LLM development but their reliability varies. We distinguish loss scaling (robust) from task scaling (unreliable).
2. **Methods:** Three scaling formulations, bootstrap CIs, AIC/BIC model selection, breakpoint detection, extrapolation analysis.
3. **Results:** Loss scaling confirms power-law with α ≈ X [CI]. Task scaling shows poor fits for Y of Z tasks. Extrapolation risk is N× larger for tasks than losses.
4. **Discussion:** Implications for compute planning, benchmark design, and the "emergent abilities" debate.
5. **Conclusion + How to Extend.**
