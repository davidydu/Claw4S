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
- Expected runtime: **8-15 minutes** on first run (324 simulations, 100K-200K rounds per matchup, parallelized across CPU cores). Runtime scales with available cores.
- All commands must be run from the **submission directory** (`submissions/pricing-collusion/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Expected: `Successfully installed numpy-2.2.4 scipy-1.15.2 matplotlib-3.10.1 pytest-8.3.5` (plus transitive deps). If pip fails, verify Python >= 3.10 with `python3 --version`.

## Step 2: Run Unit Tests

Verify the simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: `40 passed` and exit code 0. If any test fails, check that all packages from Step 1 installed correctly.

## Step 3: Run the Experiment

Execute the full pricing collusion simulation experiment:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress like `[20/324] QQ/M3/e-commerce | 0.4m elapsed | ~6m remaining`, ending with `Done. Results saved to results/` and exit code 0.

Output files created:
- `results/results.json` — 324 simulation records with auditor scores
- `results/report.md` — summary report with heatmap and statistical tests
- `results/statistical_tests.json` — per-condition statistics
- `results/figures/collusion_heatmap.png` — heatmap visualization
- `results/figures/memory_effect.png` — memory length vs collusion
- `results/figures/auditor_agreement.png` — pairwise auditor agreement

If `run.py` crashes mid-execution, check `results/progress.json` for the last completed batch.

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected output:
```
Simulations: 324
Conditions:  108
Records:     324 (expected 324)

Competitive control avg margin score: <low value near 0>

Statistical conditions: 108
Conditions with significant supra-Nash pricing (Bonferroni): N/108

Validation passed.
```

If validation fails, the error messages indicate which checks failed.

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains: collusion index heatmap (Delta by matchup × memory), auditor agreement rates, Bonferroni-corrected statistical tests, memory effect analysis, and shock robustness comparison.

## How to Extend

- **Add a pricing agent:** Subclass `BaseAgent` in `src/agents.py`, register in `AGENT_TYPES` dict, and add a matchup entry in `MATCHUPS` dict in `src/experiment.py`.
- **Add an auditor:** Subclass `BaseAuditor` in `src/auditors.py`, implement `audit(price_history, market, **kwargs)`, and add to `AuditorPanel.__init__`.
- **Add a domain preset:** Add an entry to `MARKET_PRESETS` in `src/market.py` with `n_sellers`, `alpha`, `costs`, `price_min`, `price_max`, `price_grid_size`.
- **Change market structure:** Modify the demand model in `src/market.py` (e.g., nested logit, heterogeneous consumers). Must implement `compute_demand`, `compute_profits`, `nash_price`, `monopoly_price`.
- **Add a shock type:** Add a shock class in `src/shocks.py` with `should_trigger(round)` and `apply(market)` methods, then wire into `run_simulation` in `src/experiment.py`.
