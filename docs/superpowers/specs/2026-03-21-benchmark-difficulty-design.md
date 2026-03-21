# Design Spec: Benchmark Difficulty Prediction from Structural Features

## Research Question

Can we predict which benchmark questions are hard for LLMs using only structural and information-theoretic features of the question text, without running any LLM?

## Approach

### Data Sources

1. **Easy2Hard-Bench E2H-ARC** (HuggingFace: `furonghuang-lab/Easy2Hard-Bench`, config `E2H-ARC`)
   - ARC-Challenge questions with IRT difficulty scores (0-1 scale)
   - Difficulty estimated from thousands of LLMs on the Open LLM Leaderboard
   - Source: NeurIPS 2024 Datasets & Benchmarks track
   - This is our primary dataset with ground-truth difficulty labels

2. **MMLU** (HuggingFace: `cais/mmlu`, config `all`)
   - 14,042 multiple-choice questions across 57 subjects
   - No per-question difficulty labels, but we compute structural features and use
     subject-level accuracy as a proxy for difficulty

3. **HellaSwag** (HuggingFace: `Rowan/hellaswag`)
   - 10,042 sentence completion questions
   - Used for cross-benchmark generalization analysis

### Structural Features (computed from text alone, no LLM needed)

For each question, we extract these features:

| Feature | Description | Hypothesis |
|---------|-------------|------------|
| `question_length` | Character count of question text | Longer questions may be harder |
| `word_count` | Word count of question text | More words = more complex |
| `avg_word_length` | Mean word length in characters | Technical vocabulary indicator |
| `answer_entropy` | Shannon entropy over answer option lengths | Uniform options harder to distinguish |
| `num_choices` | Number of answer choices | More choices = harder |
| `lexical_overlap` | Jaccard similarity between question and answer texts | High overlap = easier (answer in question) |
| `negation_count` | Count of negation words (not, never, except, etc.) | Negations increase difficulty |
| `question_type` | Categorical: what/which/how/why/true-false | Why/how may be harder |
| `flesch_kincaid_grade` | Readability grade level | Higher grade = harder text |
| `unique_word_ratio` | Ratio of unique words to total words | Lexical diversity |
| `max_option_length_ratio` | Ratio of longest to shortest answer option | Unbalanced options may be easier |
| `stem_overlap` | Word overlap between question stem and correct answer | More overlap = easier |

### Analysis Pipeline

1. **Data Loading**: Download E2H-ARC from HuggingFace (small text, ~2MB). Optionally load MMLU and HellaSwag for feature analysis.
2. **Feature Extraction**: Compute all 12 structural features for each question.
3. **Correlation Analysis**: Compute Spearman rank correlations between each feature and IRT difficulty.
4. **Difficulty Model**: Train a Random Forest regressor (scikit-learn) to predict difficulty from features. Use 5-fold cross-validation.
5. **Feature Importance**: Extract and rank feature importances from the model.
6. **Cross-Benchmark Analysis**: Compare feature distributions across MMLU subjects and HellaSwag.

### Key Metrics

- Spearman correlation between predicted and actual difficulty (target: rho > 0.3)
- Mean Absolute Error of difficulty prediction
- R-squared of the regression model
- Feature importance ranking
- Cross-validation stability (std of R-squared across folds)

### Fallback Strategy

If Easy2Hard-Bench is unavailable or its format has changed:
- Use ARC-Challenge directly from `allenai/ai2_arc`
- Estimate difficulty as the fraction of models that get each question wrong
  (using published accuracy data from Open LLM Leaderboard)
- Alternatively, hardcode a representative subset of ~200 questions with
  known difficulty from published IRT analyses

## Dependencies

- `numpy==2.4.3` — numerical computation
- `scipy==1.17.1` — statistical tests (Spearman correlation)
- `matplotlib==3.10.3` — plotting
- `scikit-learn==1.7.0` — Random Forest regression, cross-validation
- `datasets==4.8.3` — HuggingFace dataset loading
- `pytest==9.0.2` — testing

No PyTorch, no transformers, no model weights. Pure CPU computation.

## Expected Runtime

- Data download: ~30 seconds (small text data)
- Feature extraction: ~10 seconds
- Model training + CV: ~20 seconds
- Plotting: ~5 seconds
- Total: < 2 minutes

## Outputs

- `results/results.json` — all features, correlations, model metrics
- `results/figures/feature_correlations.png` — correlation heatmap
- `results/figures/difficulty_prediction.png` — predicted vs actual scatter
- `results/figures/feature_importance.png` — feature importance bar chart
- `results/report.md` — human-readable summary

## Scientific Hypotheses

1. **H1**: Lexical overlap between question and correct answer negatively correlates with difficulty (questions where the answer is "in the question" are easier).
2. **H2**: Answer entropy positively correlates with difficulty (uniform answer lengths make elimination harder).
3. **H3**: Negation presence increases difficulty.
4. **H4**: A Random Forest model using all structural features achieves Spearman rho > 0.3 for difficulty prediction.
5. **H5**: Feature importance ranking is stable across cross-validation folds.

## Limitations

- IRT difficulty scores are derived from LLM performance, not human performance
- Structural features cannot capture semantic reasoning difficulty
- The model may learn surface-level correlations rather than causal difficulty factors
- Cross-benchmark generalization is limited without per-question labels for MMLU/HellaSwag
