# Design Spec: Emergent Abilities -- Mirage or Real?

## 1. Research Question

Do "emergent abilities" in large language models represent genuine phase transitions in capability, or are they artifacts of discontinuous evaluation metrics? We re-analyze published benchmark data from BIG-Bench and MMLU to test the Schaeffer et al. (2023) claim that apparent emergence disappears when continuous metrics replace discontinuous ones.

## 2. Background

### 2.1 The Emergence Claim (Wei et al., 2022)
Wei et al. (arXiv:2206.07682) documented 137 tasks where LLM performance appeared to jump sharply at certain model scales. On BIG-Bench tasks like 2-digit multiplication, IPA transliteration, and word unscrambling, models showed near-zero accuracy below a threshold and then rapid improvement -- a "phase transition" pattern.

### 2.2 The Mirage Hypothesis (Schaeffer et al., 2023)
Schaeffer et al. (arXiv:2304.15004, NeurIPS 2023) argued these phase transitions are metric artifacts:
- Over 92% of claimed emergent abilities use just 2 metrics: Multiple Choice Grade and Exact String Match
- Both metrics are **discontinuous**: a near-correct answer scores identically to a random guess
- When the same model outputs are re-scored with **continuous** metrics (e.g., Token Edit Distance, Brier Score), performance improves smoothly and predictably with scale
- The "emergence" is a property of the measurement, not the model

### 2.3 Our Contribution
We re-create the core analysis from both papers using hardcoded published benchmark data, applying both discontinuous (accuracy/exact match) and continuous (token edit distance, Brier score, partial credit) metrics to the same underlying performance data. We quantify whether the "phase transition" pattern survives metric change using statistical tests for nonlinearity.

## 3. Data Sources (All Hardcoded)

### 3.1 BIG-Bench Tasks (from Wei et al. 2022, Schaeffer et al. 2023)
We hardcode published performance data for model families across scales:

**GPT-3/InstructGPT family** (parameters: 0.3B, 1.3B, 6.7B, 175B):
- 2-digit multiplication (exact string match)
- 4-digit addition (exact string match)
- Multi-step arithmetic

**LaMDA family** (parameters: 2B, 8B, 68B, 137B):
- IPA transliteration
- Word unscrambling
- Persian QA
- Swahili-English proverbs
- Word sorting
- Sports understanding

**PaLM family** (parameters: 8B, 62B, 540B):
- Arithmetic tasks
- Logical reasoning
- Language understanding

### 3.2 MMLU Scores (from Hendrycks et al. 2021, various model cards)
Overall MMLU accuracy across model sizes:
- GPT-3 family: 350M (~25.3%), 1.3B (~25.9%), 6.7B (~26.6%), 175B (43.9%)
- PaLM family: 8B (~25.3%), 62B (~53.7%), 540B (~69.3%)
- LLaMA family: 7B (~35.1%), 13B (~46.9%), 33B (~57.8%), 65B (~63.4%)
- Chinchilla: 70B (~67.5%)

### 3.3 Synthetic Ground Truth
To demonstrate the metric artifact mechanism, we generate synthetic data where:
- Per-token accuracy improves linearly with log(parameters)
- We apply both exact-match and partial-credit scoring
- This directly tests whether discontinuous metrics CREATE apparent emergence from smooth underlying improvement

## 4. Analysis Pipeline

### 4.1 Module: `src/data.py`
- Hardcoded benchmark data with inline citations
- Data structures: lists of (model_name, param_count, metric_value) tuples
- Functions to retrieve data by task, model family, metric type

### 4.2 Module: `src/analysis.py`
Core analyses:

**Analysis 1: Metric Comparison**
- For each task, plot performance vs. log(parameters) using:
  - Discontinuous metric (accuracy / exact match)
  - Continuous metric (simulated token edit distance / Brier score / partial credit)
- The continuous metric is derived from the discontinuous one using a plausible per-token error model (Schaeffer et al.'s key insight)

**Analysis 2: Nonlinearity Detection**
- Fit both linear and sigmoid (logistic) models to performance vs. log(parameters)
- Compare fits using AIC/BIC
- A task shows "true emergence" only if the sigmoid fit is significantly better than linear even with continuous metrics

**Analysis 3: Metric Sensitivity**
- For each task, compute a "Metric Sensitivity Index": ratio of apparent nonlinearity under discontinuous vs. continuous metrics
- High MSI = metric artifact; Low MSI = potentially genuine emergence

**Analysis 4: Synthetic Demonstration**
- Generate synthetic per-token accuracy that improves linearly with log(scale)
- Apply exact-match scoring (requires ALL tokens correct) vs. partial-credit
- Show that exact-match creates a phase transition from smooth underlying improvement

### 4.3 Module: `src/metrics.py`
- `exact_match_from_token_accuracy(per_token_acc, n_tokens)`: Computes P(all correct) = p^n
- `partial_credit_from_token_accuracy(per_token_acc, n_tokens)`: Computes mean(correct tokens) = p
- `brier_score(predicted_prob, true_label)`: Continuous probabilistic metric
- `token_edit_distance(per_token_acc, n_tokens)`: Expected edit distance = n*(1-p)
- `sigmoid_fit(x, y)`: Fit logistic curve, return parameters and R^2
- `linear_fit(x, y)`: Fit linear model, return parameters and R^2
- `compute_aic(n, rss, k)`: Akaike Information Criterion
- `compute_bic(n, rss, k)`: Bayesian Information Criterion

### 4.4 Module: `src/plots.py`
- `plot_metric_comparison(task_data, output_path)`: Side-by-side discontinuous vs. continuous
- `plot_nonlinearity_landscape(results, output_path)`: Heatmap of MSI across tasks
- `plot_synthetic_demo(output_path)`: The "smoking gun" -- same data, two metrics
- `plot_mmlu_scaling(output_path)`: MMLU scores across model families

### 4.5 Module: `src/report.py`
- Generate markdown summary report with key findings
- Include statistical test results

## 5. Expected Key Findings

1. **Metric artifact confirmed**: For most BIG-Bench tasks, applying continuous metrics to the same data eliminates the apparent phase transition (MSI >> 1)
2. **Some tasks may show genuine nonlinearity**: A small subset of tasks may retain sigmoid-better-than-linear fits even with continuous metrics
3. **Synthetic demonstration**: Perfectly linear per-token improvement creates dramatic "emergence" under exact-match scoring
4. **MMLU shows smooth scaling**: MMLU accuracy (already a continuous-ish metric for multiple choice) shows relatively smooth scaling across model families

## 6. Statistical Methods

- **Curve fitting**: scipy.optimize.curve_fit for sigmoid and linear models
- **Model comparison**: AIC and BIC for comparing linear vs. sigmoid fits
- **Bootstrap confidence intervals**: 1000 bootstrap resamples for fit parameters (seed=42)
- **Effect size**: Cohen's d for difference between metric types

## 7. Output Structure

```
results/
  results.json          # All numeric results
  report.md             # Summary findings
  figures/
    metric_comparison_arithmetic.png
    metric_comparison_language.png
    nonlinearity_heatmap.png
    synthetic_demo.png
    mmlu_scaling.png
```

## 8. Constraints

- **No model downloads**: All data hardcoded from published sources
- **No GPU/API calls**: Pure numerical analysis
- **Dependencies**: numpy, scipy, matplotlib, pytest only
- **Runtime**: < 2 minutes on CPU
- **Reproducibility**: seed=42 for all stochastic operations
- **All versions pinned** with ==

## 9. Limitations (to acknowledge in paper)

1. Hardcoded data points are sparse (few model sizes per family)
2. Continuous metrics are simulated from discontinuous ones using assumed per-token independence
3. The per-token independence assumption may not hold for complex reasoning tasks
4. We cannot access raw model outputs -- we work with published aggregated scores
5. Some published scores may have different evaluation protocols across papers
