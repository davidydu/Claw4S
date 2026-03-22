---
name: information-cascades
description: Simulate information cascades among Bayesian AI agents with varying sophistication. Measures cascade formation rate, accuracy, fragility, and length across 4 agent types, 3 signal qualities, and 3 sequence lengths using the BHW (1992) model. Pure Python, no external data or API keys required.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Information Cascades in AI Agent Networks

This skill simulates information cascades (herding) among sequential decision-making agents. Based on the Bikhchandani, Hirshleifer & Welch (1992) cascade model, it studies how agent sophistication affects cascade dynamics.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure computation).
- Expected runtime: **under 5 seconds** for 216 simulations.
- All commands must be run from the **submission directory** (`submissions/info-cascades/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify installation:

```bash
.venv/bin/python -c "import pytest; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify all simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: `47 passed` and exit code 0.

## Step 3: Run the Experiment

Execute the full factorial experiment (216 simulations):

```bash
.venv/bin/python run.py
```

Expected: Script prints `[4/4] Saving results to results/` and exits with code 0. Files `results/raw_results.json`, `results/metrics.json`, `results/metadata.json`, and `results/report.md` are created.

This runs:
1. 4 agent types (Bayesian, Heuristic, Contrarian, Noisy-Bayesian)
2. 3 signal qualities (q=0.6, 0.7, 0.9)
3. 3 sequence lengths (N=10, 20, 50)
4. 2 true states (A, B) for symmetry
5. 3 random seeds per condition
6. Total: 4 x 3 x 3 x 2 x 3 = 216 simulations with multiprocessing

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints simulation count (216), agent types, signal qualities, sequence lengths, symmetry check, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Cascade formation rate table (agent type x signal quality)
- Cascade accuracy table (fraction of correct cascades)
- Cascade fragility table (fraction of broken cascades)
- Mean cascade length table
- Key findings summary

## How to Extend

- **Add an agent type:** Implement the `Agent` protocol in `src/agents.py`, register it in `make_agent()`, and add it to `AGENT_TYPES` in `src/experiment.py`.
- **Change signal qualities:** Edit `SIGNAL_QUALITIES` in `src/experiment.py`.
- **Change sequence lengths:** Edit `SEQUENCE_LENGTHS` in `src/experiment.py`.
- **Add more seeds:** Extend `SEEDS` in `src/experiment.py` for tighter confidence intervals.
- **Vary contrarian rate:** Modify `p_contrarian` in `make_agent()` or parameterize it in the experiment.
