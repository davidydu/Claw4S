---
name: cascading-failures-multi-agent-networks
description: Simulate cascading failures in multi-agent AI networks. Studies how one faulty agent's errors propagate through 6 network topologies (chain, ring, star, Erdos-Renyi, scale-free, fully-connected) with 3 agent types (robust, fragile, averaging). Runs 324 simulations with multiprocessing to measure cascade size, speed, recovery time, and systemic risk.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Cascading Failures in Multi-Agent AI Networks

This skill simulates error propagation through multi-agent networks to study which topologies and agent designs are resilient vs fragile to cascading failures.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure stdlib + pytest).
- Expected runtime: **~90 seconds** for the full 324-simulation experiment.
- All commands must be run from the **submission directory** (`submissions/cascading-failures/`).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/cascading-failures/
```

All subsequent commands assume you are in this directory.

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

Verify all modules work correctly (31 tests):

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: `31 passed` and exit code 0.

## Step 3: Run Diagnostic

Quick validation with 18 simulations (1 topology, 1 agent type):

```bash
.venv/bin/python run.py --diagnostic
```

Expected: Prints report and exits with code 0. Creates `results/results.json` and `results/report.md`.

## Step 4: Run Full Experiment

Execute all 324 simulations (6 topologies x 3 agent types x 3 shock magnitudes x 2 shock locations x 3 seeds):

```bash
.venv/bin/python run.py
```

Expected: Prints `Completed 324 simulations` and full report. Creates `results/results.json` and `results/report.md`.

This will:
1. Generate networks for all 6 topologies (N=20 agents each)
2. Run paired simulations (clean baseline + shocked) for each configuration
3. Track error propagation: cascade size, speed, recovery time, systemic risk
4. Aggregate metrics across seeds with mean and standard deviation
5. Save raw and aggregated results to `results/results.json`
6. Generate summary report at `results/report.md`

## Step 5: Validate Results

Check completeness and scientific sanity:

```bash
.venv/bin/python validate.py
```

Expected: Prints simulation counts, agent comparisons, and `Validation passed.`

## Step 6: Review the Report

```bash
cat results/report.md
```

Expected: Markdown report with topology risk ranking, hub vs random attack comparison, agent type resilience ranking, and key findings.

## How to Extend

- **Add topologies:** Implement a new generator in `src/network.py` returning `AdjList`, add to `TOPOLOGIES` dict.
- **Add agent types:** Implement a new function in `src/agents.py` with signature `(List[float], float) -> float`, add to `AGENT_TYPES` dict.
- **Change parameters:** Edit `src/experiment.py` constants: `N_AGENTS`, `TOTAL_ROUNDS`, `SHOCK_MAGNITUDES`, `SEEDS`.
- **Add metrics:** Extend `src/metrics.py` with new aggregation functions.
