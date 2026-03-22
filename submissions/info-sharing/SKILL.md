---
name: info-sharing-dilemma
description: Simulate strategic information sharing among competitive AI agents. Four agent types (Open, Secretive, Reciprocal, Strategic) compete in a partial-observation environment, choosing how much private information to disclose. Sweeps 4 compositions x 3 competition levels x 3 complementarity levels x 3 seeds = 108 simulations of 10,000 rounds each. Measures sharing equilibria, group welfare, information asymmetry, and phase transitions.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Strategic Information Sharing Among Competitive AI Agents

This skill simulates the information disclosure dilemma: agents receive partial observations of a hidden state and must decide how much to share. Sharing improves group decisions but gives competitors an advantage. The experiment identifies when sharing norms emerge vs. when hoarding dominates.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure simulation).
- Expected runtime: **2-4 minutes** (108 simulations parallelized across CPU cores).
- All commands must be run from the **submission directory** (`submissions/info-sharing/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy; print(f'numpy {numpy.__version__} OK')"
```

Expected output: `numpy 2.2.4 OK`

## Step 2: Run Unit Tests

Verify the simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: 24 tests pass, exit code 0.

## Step 3: Run the Experiment

Execute the full information-sharing experiment:

```bash
.venv/bin/python run.py
```

Expected: Script prints `[3/3] Results saved to results/results.json` followed by the report, and exits with code 0. Files created: `results/results.json`, `results/analysis.json`, `results/report.md`.

This will:
1. Run 108 simulations (4 compositions x 3 competition x 3 complementarity x 3 seeds)
2. Each simulation: 4 agents play 10,000 rounds of the information-sharing game
3. Compute per-round metrics: sharing rate, group welfare, welfare gap, information asymmetry
4. Aggregate across seeds, identify phase transitions, rank agent types
5. Save raw results, statistical analysis, and a Markdown report

## Step 4: Validate Results

Check that results were produced correctly:

```bash
.venv/bin/python validate.py
```

Expected: Prints simulation counts, sanity checks, and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

The report contains:
- Agent type rankings by cumulative payoff
- Sharing rates by experimental condition (tail equilibrium, mean +/- std)
- Phase transition analysis across competition levels
- Key findings summary

## How to Extend

- **Add an agent type:** Subclass `Agent` in `src/agents.py`, register in `AGENT_TYPES`.
- **Change agent count:** Modify `n_agents` in `EnvConfig` (in `src/environment.py`) and adjust compositions in `src/experiment.py`.
- **Change competition/complementarity grid:** Edit `COMPETITION_LEVELS` and `COMPLEMENTARITY_LEVELS` in `src/experiment.py`.
- **Change round count:** Edit `N_ROUNDS` in `src/experiment.py`.
- **Change state dimensionality:** Modify `state_dim` in `EnvConfig`.
