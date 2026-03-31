# Implementation Plan: Memorization Capacity Scaling

## File Structure

```
submissions/memorization/
├── SKILL.md              # Executable skill
├── run.py                # Main entry point
├── validate.py           # Results validator
├── requirements.txt      # Pinned deps
├── conftest.py           # Pytest config
├── src/
│   ├── __init__.py
│   ├── data.py           # Synthetic data generation
│   ├── model.py          # Parameterized MLP
│   ├── train.py          # Training loop
│   ├── sweep.py          # Model size sweep
│   ├── analysis.py       # Threshold detection, sharpness
│   ├── plots.py          # Matplotlib figures
│   └── report.py         # Markdown report generation
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_train.py
│   └── test_sweep.py
├── research_note/
│   └── main.tex          # LaTeX paper (1-4 pages)
└── results/              # Generated output (gitignored)
    ├── results.json
    ├── report.md
    └── figures/
        ├── memorization_curve.png
        └── threshold_comparison.png
```

## Module Specifications

### src/data.py
- `generate_dataset(n, d, n_classes, seed, label_type)` -> (X_train, y_train, X_test, y_test)
- label_type: "random" or "structured"
- structured: k-means cluster assignments
- Returns torch tensors, deterministic with seed

### src/model.py
- `MLP(input_dim, hidden_dim, num_classes)` -> nn.Module
- `count_parameters(model)` -> int
- Simple 2-layer: Linear -> ReLU -> Linear

### src/train.py
- `train_model(model, X, y, max_epochs, lr, convergence_threshold)` -> TrainResult
- TrainResult: {final_train_acc, final_train_loss, convergence_epoch, loss_history}
- Full-batch gradient descent with Adam
- Early stopping on convergence

### src/sweep.py
- `run_sweep(hidden_dims, n, d, n_classes, seed)` -> SweepResults
- Iterates over hidden_dims, trains with both label types
- Returns structured results dict

### src/analysis.py
- `detect_threshold(params, accuracies)` -> threshold_params
- `fit_sigmoid(params, accuracies)` -> (threshold, sharpness, r_squared)
- `analyze_results(sweep_results)` -> AnalysisResults

### src/plots.py
- `plot_memorization_curves(results, output_dir)` -> saves PNG
- `plot_threshold_comparison(results, output_dir)` -> saves PNG

### src/report.py
- `generate_report(results, analysis)` -> str (markdown)
- Saves to results/report.md

## Dependencies (requirements.txt)
```
torch==2.6.0
numpy==2.2.4
scipy==1.15.2
matplotlib==3.10.1
pytest==8.3.5
```

## Key Patterns (from tokenizer-analysis)
- conftest.py: `sys.path.insert(0, os.path.dirname(__file__))`
- run.py: working-directory guard with `os.chdir()`
- validate.py: error accumulation, non-zero exit on failure
- All seeds pinned to 42
- No `source activate`, use `.venv/bin/python`
- Specific exception handling (no bare `except`)

## Runtime Estimate
- 8 widths x 2 label types = 16 training runs
- Each run: ~5-10s max (5000 epochs on 200 samples, full batch, CPU)
- Total: ~2 minutes including analysis and plotting
