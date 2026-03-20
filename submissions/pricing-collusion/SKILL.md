---
name: pricing-collusion-analysis
description: >
  Simulate algorithmic pricing agents in repeated Bertrand competition to study
  tacit collusion emergence and evaluate multi-agent auditor detection reliability
  across market conditions, memory lengths, and market shocks.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Tacit Collusion Detection in Algorithmic Pricing

This skill simulates algorithmic pricing agents competing in repeated Bertrand markets to study the emergence of tacit collusion and evaluate auditor detection reliability. The experiment sweeps over agent types, memory lengths, and market shocks, then produces a statistical report with heatmaps and auditor agreement matrices.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure simulation).
- Expected runtime: **45-60 minutes** on first run.
- All commands must be run from the **submission directory** (`submissions/pricing-collusion/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Expected: `Successfully installed numpy-... scipy-... matplotlib-... pytest-...`

## Step 2: Run Unit Tests

Verify the simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `X passed` and exit code 0.

## Step 3: Run the Experiment

Execute the full pricing collusion simulation experiment:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress for each simulation batch, ending with `[5/5] Saving results to results/` and exit code 0. Files `results/results.json`, `results/report.md`, and `results/statistical_tests.json` are created.

This will:
1. Initialize agent types (Q-learning, rule-based, mixed) across market configurations
2. Run repeated Bertrand competition simulations across memory length and shock conditions
3. Detect tacit collusion using the multi-agent auditor panel
4. Compute auditor agreement matrices and collusion heatmaps
5. Save raw results to `results/results.json` and statistical tests to `results/statistical_tests.json`
6. Generate a summary report at `results/report.md`

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints simulation count, auditor score summary, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Review the collusion heatmap, auditor agreement matrix, and key findings.

## How to Extend

- **Add a pricing agent:** Subclass `BaseAgent` in `src/agents.py` and register in the agent factory.
- **Add an auditor:** Subclass `BaseAuditor` in `src/auditors.py` and add to the auditor panel.
- **Add a domain preset:** Add an entry to `MARKET_PRESETS` in `src/market.py`.
- **Change market structure:** Modify the demand model in `src/market.py` (e.g., nested logit, heterogeneous consumers).
- **Add a shock type:** Add a shock class in `src/shocks.py` and register in the experiment runner.
