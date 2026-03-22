---
name: reward-hacking-propagation
description: Simulate how reward hacking spreads through multi-agent systems via social learning. Sweeps 324 configurations (3 initial hacker counts x 3 network topologies x 3 hack detectability levels x 4 monitor fractions x 3 seeds) across 5000 rounds with N=10 agents, measuring adoption rate, propagation speed, containment effectiveness, and welfare impact.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Reward Hacking Propagation in Multi-Agent Systems

This skill simulates how one agent's reward hack (a high-proxy-reward, low-true-reward exploit) propagates through a multi-agent system via social learning, and whether monitor agents can detect and contain the spread.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure simulation, no downloads).
- Expected runtime: **1-3 minutes** (324 simulations parallelized across CPU cores).
- All commands must be run from the **submission directory** (`submissions/reward-hacking/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy, scipy, pytest; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify all simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: 47 tests pass with exit code 0.

## Step 3: Run the Experiment

Execute the full parameter sweep (324 simulations):

```bash
.venv/bin/python run.py
```

Expected: Script prints `Completed 324 simulations.` and `Saved report to results/report.md`, then outputs the full report. Files `results/results.json` and `results/report.md` are created.

This will:
1. Build all 324 parameter combinations (3 initial hacker counts x 3 topologies x 3 detectabilities x 4 monitor fractions x 3 seeds)
2. Run each simulation for 5000 rounds with N=10 agents using multiprocessing
3. Compute summary metrics (adoption rate, propagation speed, containment, welfare)
4. Generate tables and key findings

## Step 4: Validate Results

Check that results are complete and scientifically sound:

```bash
.venv/bin/python validate.py
```

Expected: Prints simulation/entry counts and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Steady-state adoption rate by topology and monitor fraction
- Propagation speed by initial hacker count and topology
- Containment effectiveness by detectability and monitor fraction
- Welfare impact (proxy-true reward divergence) by topology and monitor fraction
- Key findings summary

## How to Extend

- **Change agent count:** Modify `N_AGENTS` in `src/experiment.py`.
- **Add a topology:** Add a builder in `src/network.py` and register it in `build_adjacency()`.
- **Change reward parameters:** Edit `HACK_PROXY_MEAN`, `HONEST_PROXY_MEAN`, etc. in `src/agents.py`.
- **Add an agent type:** Define a new `BETA_*` constant in `src/agents.py` and handle it in `create_agent_population()`.
- **Change detectability levels:** Edit `DETECT_THRESHOLDS` in `src/simulation.py`.
- **Add a metric:** Extend `compute_summary_metrics()` in `src/metrics.py`.
