# Skill: Adversarial World Model Manipulation

Reproduce the experiments from "How Fast Can You Break a World Model?
Adversarial Belief Manipulation in Multi-Agent Systems."

A repeated signaling game where adversaries strategically send
misleading signals to corrupt a Bayesian learner's world model.
We measure belief distortion, manipulation speed, decision quality,
and credibility exploitation across 162 simulations.

## Prerequisites

- Python 3.11+
- ~200 MB disk for results (figures, JSON, pickle)
- ~16 seconds wall-clock on an 8-core machine

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/world-model-adversarial/
```

All subsequent commands assume you are in this directory.

## Step 1: Create virtual environment and install dependencies

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

**Expected output:** `Successfully installed numpy-2.2.4 scipy-1.15.2 matplotlib-3.10.1 pytest-8.3.5 ...`

## Step 2: Run tests (62 tests)

```bash
.venv/bin/python -m pytest tests/ -v
```

**Expected output:** `62 passed` with 0 failures. Tests cover:
- `test_environment.py` (9 tests): state drift, noisy signals, reset
- `test_agents.py` (23 tests): belief updates, trust dynamics, factories
- `test_auditors.py` (12 tests): distortion, credibility, decision quality, recovery
- `test_experiment.py` (10 tests): simulation runner, reproducibility, experiment matrix
- `test_integration.py` (8 tests): end-to-end simulation, metric ordering

## Step 3: Run the experiment

```bash
.venv/bin/python run.py --n-rounds 50000 --seeds 0,1,2
```

**Expected output:**
- `162/162` simulations completed
- Runtime ~15 seconds
- `results/` directory created with:
  - `summary.json` (54 aggregate groups)
  - `raw_results.pkl` (162 simulation results)
  - `manipulation_speed.json`
  - `figures/` (32 PNG files: heatmaps, time series, bar charts)
  - `tables/` (6 CSV files: distortion, accuracy, resilience)

## Step 4: Validate results

```bash
.venv/bin/python validate.py
```

**Expected output:** `15/15 checks passed` covering:
- 162 simulations completed
- 54 aggregate groups in summary
- SA distorts more than RA for all learners
- SL more resilient than NL in some regime
- PA exploitation pattern detected
- All belief errors in [0, 1]
- Reproducibility (re-run 2 configs, diff < 1e-10)

## Key Results

| Matchup    | Stable Err | Volatile Err | Stable Acc | Volatile Acc |
|------------|-----------|-------------|-----------|-------------|
| NL-vs-RA   | 0.806     | 0.794       | 0.202     | 0.191       |
| NL-vs-SA   | 0.998     | 0.995       | 0.000     | 0.002       |
| SL-vs-SA   | 0.998     | 0.992       | 0.000     | 0.004       |
| AL-vs-SA   | 0.998     | 0.995       | 0.000     | 0.002       |
| AL-vs-RA   | 0.811     | 0.769       | 0.189     | 0.204       |

Main findings:
1. Strategic adversaries (SA) achieve near-total belief distortion (0.998) against all learner types in stable environments.
2. Volatile environments create small but consistent resilience advantages for skeptical (SL) and adaptive (AL) learners.
3. The Patient Adversary (PA) shows a clear credibility exploitation pattern detectable by the auditor.
4. Signal-action trust (used by AL) cannot detect deception when deceptive signals are consistent -- a fundamental limitation of trust-based defenses.

## How to Extend

### Add a new learner type
1. Subclass `Learner` in `src/agents.py`.
2. Implement `update(signal)` with your update rule.
3. Add to `LEARNER_TYPES` dict with a 2-letter code.
4. Add tests in `tests/test_agents.py`.
5. Re-run: the experiment matrix auto-includes new learner codes.

### Add a new adversary type
1. Subclass `Adversary` in `src/agents.py`.
2. Implement `choose_signal(true_state, learner_beliefs)`.
3. Add to `ADVERSARY_TYPES` dict.
4. Add tests and re-run.

### Change environment parameters
- **States:** `--n-states N` (default 5) in `SimConfig`
- **Drift intervals:** Modify `_DEFAULT_DRIFT_INTERVALS` in `src/environment.py`
- **Signal noise:** Already parameterized (0.0 and 0.1)

### Add a new auditor
1. Create a class with an `audit(trace: SimTrace) -> dict[str, float]` method.
2. Add to `ALL_AUDITORS` in `src/auditors.py`.
3. Metrics auto-propagate to summary tables.

### Adapt to a different domain
The framework generalizes to any setting where:
- An agent maintains beliefs about a hidden state
- Another agent can send signals to influence those beliefs
- You want to measure the effectiveness of manipulation

Examples: financial market manipulation, propaganda spread, adversarial NLP.
