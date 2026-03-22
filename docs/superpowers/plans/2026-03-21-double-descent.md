# Double Descent Implementation Plan

## Task List

### Task 1: Project Scaffolding
- Create `submissions/double-descent/` with: `src/__init__.py`, `tests/__init__.py`, `conftest.py`, `requirements.txt`
- requirements.txt: torch==2.6.0, numpy==2.2.4, scipy==1.15.2, matplotlib==3.10.1, pytest==8.3.5
- Create venv with `/opt/homebrew/bin/python3.13 -m venv .venv`
- Install deps

### Task 2: Data Generation (`src/data.py`)
- `generate_regression_data(n_train, n_test, d, noise_std, seed)` -> (X_train, y_train, X_test, y_test)
- X ~ N(0,1), y = X @ w_true + noise
- Normalize X columns to zero mean, unit variance
- Return torch tensors
- Test: `tests/test_data.py` — shapes, determinism, noise levels

### Task 3: Model (`src/model.py`)
- `class MLP(nn.Module)`: 2-layer MLP with configurable hidden width
- `count_parameters(model)` -> int
- `get_interpolation_threshold(n_train, d)` -> approximate width
- Test: `tests/test_model.py` — param count formula, forward pass shapes

### Task 4: Training (`src/training.py`)
- `train_model(model, X_train, y_train, X_test, y_test, epochs, lr, record_every)` -> dict with train_losses, test_losses per epoch
- SGD optimizer, MSE loss, no regularization
- Record metrics at intervals specified by record_every
- Test: `tests/test_training.py` — loss decreases, output structure

### Task 5: Sweep (`src/sweep.py`)
- `model_wise_sweep(widths, n_train, n_test, d, noise_std, epochs, lr, seed)` -> list of dicts
- `epoch_wise_sweep(width, n_train, n_test, d, noise_std, max_epochs, lr, record_every, seed)` -> dict
- `phase_diagram_sweep(widths, n_train, n_test, d, noise_std, max_epochs, lr, record_every, seed)` -> list of dicts
- Test: `tests/test_sweep.py` — output structure, small smoke test

### Task 6: Analysis (`src/analysis.py`)
- `find_interpolation_peak(sweep_results)` -> (peak_width, peak_test_loss)
- `compute_double_descent_ratio(sweep_results)` -> ratio of peak to minimum
- `detect_epoch_wise_double_descent(epoch_results)` -> bool
- Test: `tests/test_analysis.py`

### Task 7: Plotting (`src/plots.py`)
- `plot_model_wise(results, output_path)` — test MSE vs width, with interpolation threshold line
- `plot_epoch_wise(results, output_path)` — test MSE vs epoch
- `plot_noise_comparison(results_by_noise, output_path)` — overlay curves
- `plot_phase_diagram(results, output_path)` — 2D heatmap
- No tests (visual output)

### Task 8: Report Generation (`src/report.py`)
- `generate_report(all_results)` -> markdown string
- Summarize findings, peak location, double descent ratio, noise effect

### Task 9: run.py
- Working directory guard (must run from submissions/double-descent/)
- Orchestrate: data gen -> model-wise sweep (3 noise levels) -> epoch-wise sweep -> phase diagram -> plots -> report
- Save results/results.json, all plots, results/report.md
- Print progress with step numbers

### Task 10: validate.py
- Load results.json, check structure
- Verify double descent detected (peak exists)
- Verify plots exist
- Verify runtime metadata

### Task 11: SKILL.md
- Follow tokenizer-analysis pattern
- Steps: setup, tests, run, validate, review

### Task 12: Research Note (`research_note/main.tex`)
- 2-3 pages, authors: Yun Du, Lina Ji, Claw
- Introduction, Methods, Results, Discussion
- Reference Nakkiran 2019, Belkin 2019, Advani & Saxe 2017

### Task 13: E2E Test & Polish
- Delete .venv and results, re-run from scratch
- Fix any issues
- Self-score against rubric
