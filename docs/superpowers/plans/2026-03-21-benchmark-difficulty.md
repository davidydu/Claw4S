# Implementation Plan: Benchmark Difficulty Prediction

## Task List

### Task 1: Scaffold submission directory
- Create `submissions/benchmark-difficulty/` with `__init__.py`, `conftest.py`
- Create `src/__init__.py`
- Create `tests/__init__.py`
- Create `requirements.txt` with pinned deps
- Create venv and install deps

### Task 2: Data module (`src/data.py`)
- TDD: Write `tests/test_data.py` first
  - Test `load_arc_with_difficulty()` returns list of dicts with required keys
  - Test each dict has: question, choices, answer, difficulty
  - Test difficulty values are in [0, 1]
  - Test at least 100 questions loaded
- Implement: Download E2H-ARC from HuggingFace `furonghuang-lab/Easy2Hard-Bench`
- Fallback: If E2H-ARC unavailable, hardcode ~200 ARC questions with synthetic difficulty
- Commit

### Task 3: Feature extraction module (`src/features.py`)
- TDD: Write `tests/test_features.py` first
  - Test `extract_features(question_dict)` returns dict with all 12 feature keys
  - Test feature values are numeric and finite
  - Test edge cases: empty question, single-word question, single choice
  - Test `extract_all_features(questions)` returns list of feature dicts
- Implement all 12 structural features
- Commit

### Task 4: Analysis module (`src/analysis.py`)
- TDD: Write `tests/test_analysis.py` first
  - Test `compute_correlations(features, difficulties)` returns dict of Spearman rhos
  - Test `train_difficulty_model(features, difficulties)` returns model with R² > 0
  - Test `cross_validate_model(features, difficulties)` returns CV scores
  - Test model predictions are in [0, 1] range
- Implement: Spearman correlations, Random Forest regression, 5-fold CV
- Commit

### Task 5: Plotting module (`src/plots.py`)
- TDD: Write `tests/test_plots.py` first
  - Test each plot function creates a file at the expected path
  - Test plots don't raise exceptions
- Implement 3 plots:
  - `plot_feature_correlations()` — bar chart of Spearman rhos
  - `plot_difficulty_prediction()` — scatter of predicted vs actual
  - `plot_feature_importance()` — bar chart of RF feature importances
- Commit

### Task 6: Report module (`src/report.py`)
- TDD: Write `tests/test_report.py` first
  - Test `generate_report(results)` returns non-empty string
  - Test report contains key sections
- Implement: Generate markdown report from results dict
- Commit

### Task 7: Orchestration (`run.py`, `validate.py`)
- Implement `run.py`: load data, extract features, analyze, plot, save results
- Implement `validate.py`: check results.json exists, validate metrics
- Test E2E: `run.py` then `validate.py`
- Commit

### Task 8: SKILL.md
- Write SKILL.md following tokenizer-analysis pattern
- Include prerequisites, steps, expected outputs, how to extend
- Commit

### Task 9: Research note
- Write `research_note/main.tex` (1-4 pages)
- Authors: Yun Du, Lina Ji, Claw
- Commit

### Task 10: Final review and polish
- Run full test suite
- Run E2E from clean state
- Self-score against rubric
- Fix any issues
- Final commit
