---
name: emergent-social-norms
description: Simulate emergent social norms in AI agent populations playing coordination games. Measures norm convergence time, efficiency, diversity, and fragility across 4 population compositions, 3 game structures, 3 population sizes, and 3 seeds (108 simulations total, 50k pairwise interactions each). Uses multiprocessing for parallelism.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Emergent Social Norms in AI Agent Populations

This skill simulates how behavioral conventions (norms) emerge among heterogeneous AI agent populations interacting in repeated coordination games — without explicit coordination.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure simulation).
- Expected runtime: **5-8 minutes** (108 simulations parallelized across CPU cores).
- All commands must be run from the **submission directory** (`submissions/emergent-norms/`).

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy; print('All imports OK')"
```

Expected output: `All imports OK`

## Step 2: Run Unit Tests

Verify the simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: **38 passed** and exit code 0.

## Step 3: Run the Experiment

Execute the full emergent norms experiment (108 simulations):

```bash
.venv/bin/python run.py
```

Expected: Script prints `Done. Output saved to results/results.json and results/report.md` and exits with code 0. The experiment sweeps over:
- 4 population compositions (all_adaptive, mixed_conform, innovator_heavy, traditionalist_heavy)
- 3 game structures (symmetric, asymmetric, dominant equilibrium)
- 3 population sizes (N=20, 50, 100)
- 3 random seeds (42, 123, 7)

Each simulation runs 50,000 pairwise interactions. Four metrics are computed per simulation:
1. **Norm convergence time** — round at which one action captures >=80% of a trailing window
2. **Norm efficiency** — ratio of realized payoff to optimal coordination payoff
3. **Norm diversity** — number of behavioral clusters (actions with >=10% share)
4. **Norm fragility** — fraction of innovators needed to displace the dominant norm

## Step 4: Validate Results

Check that results are complete and all metrics are in valid ranges:

```bash
.venv/bin/python validate.py
```

Expected: Prints simulation counts, metric summaries, and `Validation passed.`

## Step 5: Review the Report

Read the generated summary report:

```bash
cat results/report.md
```

Expected: A markdown report with 4 sections covering convergence by composition, efficiency by game structure, scale effects, and key findings.

## How to Extend

- **New agent types:** Add a new `AgentType` enum value in `src/agents.py` and implement its `choose_action` logic. Add the type to compositions in `src/experiment.py`.
- **New game structures:** Add a `make_*_game()` function in `src/game.py` and register it in `ALL_GAMES`.
- **More population sizes or seeds:** Edit `POPULATION_SIZES` and `SEEDS` in `src/experiment.py`.
- **Different interaction counts:** Pass `total_rounds=N` to `run_experiment()` in `run.py`.
- **New metrics:** Add a function to `src/metrics.py` and call it from `compute_sim_metrics()` in `src/simulation.py`.
