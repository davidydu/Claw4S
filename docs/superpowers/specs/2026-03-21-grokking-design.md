# Design Spec: Grokking Phase Diagrams

## 1. Problem Statement

"Grokking" is a phenomenon where neural networks trained on algorithmic tasks exhibit delayed generalization: training accuracy reaches near-perfect performance long before test accuracy improves. This creates a phase transition from memorization to generalization. The phenomenon was first documented by Power et al. (2022) on modular arithmetic tasks.

We aim to systematically map the **phase diagram** of grokking as a function of three hyperparameters: weight decay, dataset fraction, and model width. This reveals the boundaries between four regimes: (1) confusion (neither memorizes nor generalizes), (2) memorization only, (3) grokking (delayed generalization), and (4) comprehension (fast generalization).

## 2. Scientific Background

### 2.1 Key References
- **Power et al. 2022**: Original grokking paper. Showed delayed generalization on modular arithmetic with small transformers. Key finding: weight decay is critical for grokking.
- **Nanda et al. 2023**: Mechanistic interpretability of grokking. Reverse-engineered the learned algorithm as discrete Fourier transform + trigonometric identities. Identified three training phases: memorization, circuit formation, cleanup.
- **Liu et al. 2022 (Omnigrok)**: Extended grokking beyond algorithmic data. Introduced "LU mechanism" — mismatch between L-shaped training loss and U-shaped test loss as function of weight norm. Showed grokking depends on initialization scale and weight decay.

### 2.2 Four Phases of Learning
Following Liu et al. (Omnigrok), we classify each training run into one of four outcomes:
1. **Confusion**: Neither train nor test accuracy reaches 95%. Model fails to learn.
2. **Memorization**: Train accuracy > 95% but test accuracy < 95% at end of training. Overfits without generalizing.
3. **Grokking**: Train accuracy reaches 95% first, then test accuracy reaches 95% later (delayed generalization). The gap between these epochs is the "grokking gap."
4. **Comprehension**: Both train and test accuracy reach 95%, with test following quickly (within 200 epochs of train). Fast generalization without grokking delay.

## 3. Experimental Design

### 3.1 Task: Modular Addition
- Task: Given (a, b), predict (a + b) mod p
- Prime p = 97 (standard in literature)
- Full dataset: all p^2 = 9409 pairs
- Input encoding: one-hot embeddings for a and b (vocabulary size p each)
- Output: classification over p classes

### 3.2 Model: Tiny MLP
- Architecture: Embedding(p, d_embed) for each input, concatenate, MLP with 1 hidden layer, output p classes
- Embedding dimension: fixed at 16
- Hidden dimension: variable (sweep parameter), values: [16, 32, 64]
- Activation: ReLU
- Total parameters: roughly 2*p*d_embed + d_embed*2*d_hidden + d_hidden*p
  - For d_hidden=32: ~2*97*16 + 32*32 + 32*97 = 3104 + 1024 + 3104 = ~7.2K params
  - For d_hidden=128: ~2*97*16 + 32*128 + 128*97 = 3104 + 4096 + 12416 = ~19.6K params
  - All well under 100K limit

### 3.3 Training Configuration
- Optimizer: AdamW (lr=1e-3, betas=(0.9, 0.98))
- Loss: Cross-entropy
- Batch size: full batch (all training examples in one batch) — standard for grokking studies
- Max epochs: 5000
- Device: CPU only (torch.device("cpu"))
- Seeds: all pinned to 42

### 3.4 Phase Diagram Sweep
Three-dimensional sweep:
- **Weight decay**: [0.0, 0.001, 0.01, 0.1, 1.0] — 5 values
- **Dataset fraction** (fraction of p^2 pairs used for training): [0.3, 0.5, 0.7, 0.9] — 4 values
- **Hidden dimension**: [32, 64, 128] — 3 values

Total: 5 * 4 * 3 = 60 training runs

### 3.5 Runtime Budget
- Each run: ~5000 epochs on ~3000-8000 examples with a <20K param model
- Estimated time per run: 2-4 seconds on CPU
- Total: 60 runs * ~3 seconds = ~180 seconds = 3 minutes
- Safety margin: reduce max_epochs or early-stop if needed

## 4. Metrics & Grokking Detection

### 4.1 Per-Run Metrics
- Train accuracy (per epoch, logged every 100 epochs + final)
- Test accuracy (per epoch, logged every 100 epochs + final)
- Train loss (cross-entropy)
- Test loss (cross-entropy)
- Epoch where train accuracy first exceeds 95% (epoch_train_95)
- Epoch where test accuracy first exceeds 95% (epoch_test_95)
- Grokking gap: epoch_test_95 - epoch_train_95 (if both achieved)

### 4.2 Phase Classification
For each run, classify into one of four phases:
- **Confusion**: final train_acc < 95% AND final test_acc < 95%
- **Memorization**: final train_acc >= 95% AND final test_acc < 95%
- **Grokking**: epoch_train_95 exists AND epoch_test_95 exists AND grokking_gap > 200
- **Comprehension**: epoch_train_95 exists AND epoch_test_95 exists AND grokking_gap <= 200

### 4.3 Aggregate Outputs
- Phase diagram heatmaps (2D slices): weight_decay x dataset_fraction for each hidden_dim
- Grokking gap heatmaps (where grokking occurs)
- Summary statistics: fraction of runs in each phase, mean grokking gap

## 5. Output Artifacts

### 5.1 Data Files (results/)
- `results/sweep_results.json`: Full sweep results with per-run metrics
- `results/phase_diagram.json`: Phase classifications for all 60 runs
- `results/report.md`: Human-readable summary report

### 5.2 Plots (results/)
- `results/phase_diagram_h32.png`: Phase heatmap for hidden_dim=32
- `results/phase_diagram_h64.png`: Phase heatmap for hidden_dim=64
- `results/phase_diagram_h128.png`: Phase heatmap for hidden_dim=128
- `results/grokking_curves.png`: Example training curves showing grokking

## 6. Module Structure

```
submissions/grokking/
├── SKILL.md
├── run.py
├── validate.py
├── requirements.txt
├── conftest.py
├── src/
│   ├── __init__.py
│   ├── data.py          # Modular arithmetic dataset generation
│   ├── model.py         # Tiny MLP model
│   ├── train.py         # Training loop with metric logging
│   ├── sweep.py         # Phase diagram sweep orchestration
│   ├── analysis.py      # Phase classification and analysis
│   ├── plots.py         # Matplotlib visualization
│   └── report.py        # Markdown report generation
├── tests/
│   ├── __init__.py
│   ├── test_data.py     # Test data generation
│   ├── test_model.py    # Test model architecture
│   ├── test_train.py    # Test training loop
│   ├── test_analysis.py # Test phase classification
│   └── test_sweep.py    # Test sweep logic
└── research_note/
    └── main.tex
```

## 7. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Training too slow on CPU | Exceeds 3-min budget | Reduce max_epochs, use early stopping, reduce sweep points |
| No grokking observed | No interesting results | Weight decay range covers known grokking region; p=97 is validated in literature |
| Non-deterministic results | Reproducibility fails | Pin all seeds (torch, numpy), use deterministic operations |
| Memory issues | Crash | Models are <20K params, datasets <10K examples — negligible memory |
