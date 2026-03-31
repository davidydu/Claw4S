# Implementation Plan: Emergent Abilities -- Mirage or Real?

## Task Sequence (TDD: test -> verify fail -> implement -> verify pass -> commit)

### Task 1: Scaffold submission directory
- Create `submissions/emergent-abilities/` with empty files: `__init__.py`, `conftest.py`, `requirements.txt`
- Create `src/__init__.py`, `tests/__init__.py`
- Create venv with `/opt/homebrew/bin/python3.13 -m venv .venv`
- Install deps: numpy==2.2.3, scipy==1.15.2, matplotlib==3.10.1, pytest==8.3.5
- Commit

### Task 2: Implement `src/data.py` with hardcoded benchmark data
- Write `tests/test_data.py`:
  - test_bigbench_tasks_exist: verify >= 4 tasks
  - test_bigbench_data_structure: each entry has model, params, accuracy
  - test_mmlu_data_exists: verify >= 3 model families
  - test_param_counts_positive: all param counts > 0
  - test_accuracy_in_range: all accuracy values in [0, 1]
- Run tests -> verify fail
- Implement `src/data.py` with hardcoded data + inline citations
- Run tests -> verify pass
- Commit

### Task 3: Implement `src/metrics.py` with scoring functions
- Write `tests/test_metrics.py`:
  - test_exact_match_perfect: p=1.0 -> 1.0
  - test_exact_match_zero: p=0.0 -> 0.0
  - test_exact_match_partial: 0 < p < 1 -> p^n
  - test_partial_credit: equals p regardless of n
  - test_token_edit_distance: equals n*(1-p)
  - test_brier_score: known values
  - test_sigmoid_fit: recovers known sigmoid parameters
  - test_linear_fit: recovers known linear parameters
  - test_aic_bic: known values
- Run tests -> verify fail
- Implement `src/metrics.py`
- Run tests -> verify pass
- Commit

### Task 4: Implement `src/analysis.py` with core analyses
- Write `tests/test_analysis.py`:
  - test_metric_comparison_returns_results: has both metric types
  - test_nonlinearity_detection: sigmoid fits better for known sigmoid data
  - test_msi_high_for_artifact: synthetic artifact data has MSI > 2
  - test_synthetic_demo_shows_divergence: exact match << partial credit at low p
  - test_analysis_deterministic: same seed -> same results
- Run tests -> verify fail
- Implement `src/analysis.py`
- Run tests -> verify pass
- Commit

### Task 5: Implement `src/plots.py` with visualization
- Write `tests/test_plots.py`:
  - test_plot_metric_comparison_creates_file
  - test_plot_synthetic_demo_creates_file
  - test_plot_nonlinearity_heatmap_creates_file
  - test_plot_mmlu_scaling_creates_file
  - test_plots_are_png
- Run tests -> verify fail
- Implement `src/plots.py`
- Run tests -> verify pass
- Commit

### Task 6: Implement `src/report.py` with markdown report generation
- Write `tests/test_report.py`:
  - test_report_contains_title
  - test_report_contains_findings
  - test_report_contains_methodology
  - test_report_contains_limitations
- Run tests -> verify fail
- Implement `src/report.py`
- Run tests -> verify pass
- Commit

### Task 7: Implement `run.py` and `validate.py`
- Implement `run.py`: orchestrator that runs analysis, generates plots, writes report
- Implement `validate.py`: checks results/ for completeness and correctness
- Run full pipeline: `run.py` then `validate.py`
- Commit

### Task 8: Write SKILL.md
- Follow pattern from tokenizer-analysis submission
- Include prerequisites, steps, expected outputs, how to extend
- Commit

### Task 9: Write research note (LaTeX)
- Create `research_note/main.tex`
- Authors: Yun Du, Lina Ji, Claw
- 1-4 pages covering: introduction, methods, results, discussion, limitations
- Commit

### Task 10: E2E validation and polish
- Delete .venv and results/, re-run from scratch following SKILL.md
- Run full test suite
- Fix any issues
- Final commit
