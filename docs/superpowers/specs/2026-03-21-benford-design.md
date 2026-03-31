# Design Spec: Benford's Law in Trained Neural Networks

**Date:** 2026-03-21
**Authors:** Yun Du, Lina Ji, Claw
**Submission:** Claw4S 2026

## 1. Research Question

Do the leading digits of trained neural network weight values follow Benford's Law? How does conformity to Benford's Law vary across:
- Training progress (random init vs. partially trained vs. fully converged)
- Network layers (input-adjacent vs. output-adjacent)
- Model size (hidden dimension 64 vs. 128)
- Task type (modular arithmetic vs. regression)

## 2. Background

**Benford's Law** states that in many naturally occurring datasets, the leading significant digit d (1-9) appears with probability:

    P(d) = log10(1 + 1/d)

This gives digit 1 a ~30.1% frequency, digit 2 ~17.6%, down to digit 9 at ~4.6%.

**Prior work** (Sahu 2021, "Rethinking Neural Networks with Benford's Law") introduced MLH (Model Enthalpy) measuring closeness of weight distributions to Benford's Law, showing correlation with generalization. Toosi 2025 confirmed this in RNNs/LSTMs. Research shows:
- Trained weights tend toward Benford conformity regardless of initialization
- Conformity increases with training and correlates with model performance
- This holds across architectures (AlexNet to ResNeXt to Transformers)

**Our contribution:** A fully reproducible, agent-executable study of Benford's Law emergence in tiny MLPs, tracking conformity per-layer and across training with rigorous statistical testing (chi-squared, MAD). Unlike prior work on large models, we use minimal MLPs trainable on CPU in under 3 minutes, making the experiment fully reproducible by any agent.

## 3. Experimental Design

### 3.1 Tasks
1. **Modular arithmetic (mod 97):** Input (a, b) encoded as one-hot or scalar, predict (a + b) mod 97. Known to exhibit "grokking" -- sudden generalization after memorization.
2. **Sine regression:** Input x in [0, 2*pi], predict sin(x). Smooth function approximation task.

### 3.2 Models
Tiny MLPs with ReLU activation:
- **Small:** Input -> Linear(d_in, 64) -> ReLU -> Linear(64, 64) -> ReLU -> Linear(64, d_out)
- **Large:** Input -> Linear(d_in, 128) -> ReLU -> Linear(128, 128) -> ReLU -> Linear(128, d_out)

### 3.3 Training Protocol
- Optimizer: Adam, lr=1e-3
- Seed: 42 (deterministic)
- Epochs: up to 5000 (or early stopping on convergence)
- Weight snapshots saved at: epoch 0 (init), 100, 500, 1000, 2000, 5000

### 3.4 Benford Analysis
For each weight snapshot:
1. Extract all weight values from each layer (exclude biases for primary analysis, include in secondary)
2. Take absolute values, exclude zeros and values < 1e-10
3. Extract leading significant digit: d = int(str(|w| in scientific notation)[0])
4. Count frequency of each digit 1-9
5. Compare observed distribution to Benford's expected distribution

### 3.5 Statistical Tests
1. **Chi-squared test:** chi2 = sum_d N * (observed_d - expected_d)^2 / expected_d, with 8 degrees of freedom. p < 0.05 rejects Benford conformity.
2. **MAD (Mean Absolute Deviation):** MAD = (1/9) * sum_d |observed_d - expected_d|. Nigrini's thresholds: < 0.006 = close conformity, 0.006-0.012 = acceptable, 0.012-0.015 = marginal, > 0.015 = nonconformity.
3. **KL divergence** from Benford distribution (supplementary metric).

### 3.6 Controls
- **Random initialization (epoch 0):** Baseline -- do randomly initialized weights follow Benford?
- **Uniform random:** Weights drawn from U(-1, 1) -- should NOT follow Benford.
- **Normal random:** Weights drawn from N(0, 0.01) -- may weakly conform.

## 4. Expected Findings

Based on prior literature:
1. Randomly initialized weights (Kaiming/Xavier) will show partial Benford conformity due to the log-normal-like tail of their distributions.
2. Trained weights will show stronger conformity than random initialization.
3. Deeper layers may show different conformity patterns than shallow layers.
4. Larger models may show stronger conformity due to more parameters.
5. Uniform random control will show poor Benford conformity.

## 5. Deliverables

- `SKILL.md`: Agent-executable instructions
- `run.py`: Main analysis runner
- `validate.py`: Results validator
- `src/data.py`: Task data generation
- `src/model.py`: MLP definition
- `src/train.py`: Training with snapshot saving
- `src/benford_analysis.py`: Leading digit extraction, Benford comparison, statistical tests
- `src/plots.py`: Visualization generation
- `src/report.py`: Markdown report generation
- `tests/`: Unit tests for all modules
- `research_note/main.tex`: 1-4 page LaTeX paper

## 6. Runtime Budget

Target: <= 3 minutes total on CPU.
- Training 4 models (2 tasks x 2 sizes) x 5000 epochs each: ~2 minutes
- Analysis and report generation: ~30 seconds
- Tests: ~30 seconds

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Training too slow on CPU | Use tiny models (64/128 hidden), short epochs, reduce if needed |
| Weights don't follow Benford | This IS a finding -- document deviation patterns |
| Too few weights per layer for chi-squared | Pool weights across layers for aggregate test, report per-layer with caveat |
| Non-determinism | Pin seed=42 everywhere, use torch.manual_seed, set torch deterministic mode |
