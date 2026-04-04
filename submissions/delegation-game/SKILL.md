---
name: delegation-game
description: Simulate strategic delegation in AI agent hierarchies using a principal-agent model. Compares 4 incentive schemes (fixed-pay, piece-rate, tournament, reputation-based) across 4 worker compositions, 3 noise levels, and 3 seeds (144 simulations, 10k rounds each). Measures quality, shirking rate, principal payoff, worker surplus, and incentive efficiency.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# The Delegation Dilemma: When AI Agents Outsource Decisions to Sub-Agents

This skill runs a principal-agent simulation studying how different incentive structures affect worker behavior when a principal delegates tasks to worker agents under moral hazard (unobservable effort).

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure simulation).
- Expected runtime: **30-60 seconds** (144 simulations with multiprocessing).
- All commands must be run from the **submission directory** (`submissions/delegation-game/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy, pytest; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify all simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: 39 tests pass, exit code 0.

## Step 3: Run the Experiment

Execute the full 144-simulation sweep:

```bash
.venv/bin/python run.py
```

Expected: Prints `[3/3] Saving results to results/` and the full Markdown report. Files `results/results.json` and `results/report.md` are created.

This will:
1. Build the 144-configuration grid (4 schemes x 4 compositions x 3 noise levels x 3 seeds)
2. Run all simulations in parallel using multiprocessing (10,000 rounds each)
3. Aggregate results across seeds (mean and std)
4. Generate a summary report

## Step 4: Validate Results

Check that results are complete and internally consistent:

```bash
.venv/bin/python validate.py
```

Expected: Prints simulation counts, behavioral checks, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Average quality tables by scheme and worker composition for each noise level
- Incentive efficiency tables (quality per dollar spent)
- Shirking rate tables
- Key findings summary

## How to Extend

- **Add a worker type:** Implement the `Worker` protocol in `src/workers.py` and register in `create_worker()`.
- **Add an incentive scheme:** Subclass `IncentiveScheme` in `src/incentives.py` and register in `SCHEME_REGISTRY`.
- **Change the grid:** Modify `WORKER_COMPOSITIONS`, `NOISE_LEVELS`, or `SEEDS` in `src/experiment.py`.
- **Change simulation length:** Adjust `NUM_ROUNDS` in `src/experiment.py`.
- **Add metrics:** Extend `SimResult` in `src/simulation.py` and the aggregation in `src/experiment.py`.
