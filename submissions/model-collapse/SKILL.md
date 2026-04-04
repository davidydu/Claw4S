---
name: model-collapse-multi-agent
description: Simulate model collapse in multi-agent data ecosystems where AI agents train on each other's outputs across generations. Measures KL divergence from ground truth for 3 agent types (naive, selective, anchored) across 5 ground-truth fractions, 3 distributions, and 3 seeds (135 simulations). Identifies collapse thresholds, curve shapes, and minimum ground-truth anchoring needed to prevent quality degradation.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Model Collapse in Multi-Agent Data Ecosystems

This skill simulates iterative model collapse: agents learn distributions from training data, produce synthetic data, and the next generation trains on that synthetic output. Over generations, quality (measured by KL divergence from ground truth) degrades -- unless ground-truth data is mixed in.

## Prerequisites

- Requires **Python 3.10+**. No internet access or API keys needed.
- Expected runtime: **~90 seconds** (8-core parallel).
- All commands must be run from the **submission directory** (`submissions/model-collapse/`).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/model-collapse/
```

All subsequent commands assume you are in this directory.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify all packages are installed:

```bash
.venv/bin/python -c "import numpy, scipy; print(f'numpy={numpy.__version__} scipy={scipy.__version__}')"
```

Expected output: `numpy=2.4.3 scipy=1.17.1`

## Step 2: Run Unit Tests

Verify all modules work correctly (41 tests):

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: `41 passed` with exit code 0.

## Step 3: Run the Experiment

Execute the full 135-simulation grid:

```bash
.venv/bin/python run.py
```

Expected: Script prints `[3/3] Generating report...` and exits with code 0. Creates `results/results.json`, `results/summary.json`, and `results/report.md`.

This runs:
1. 3 agent types (naive, selective, anchored) x 5 GT fractions (0%, 1%, 5%, 10%, 50%) x 3 distributions (bimodal, skewed, uniform-like) x 3 seeds = 135 simulations
2. Each simulation runs 10 generational iterations
3. All simulations execute in parallel via multiprocessing

## Step 4: Validate Results

Check completeness and scientific soundness:

```bash
.venv/bin/python validate.py
```

Expected: 7 checks all print `[OK]`, ending with `Validation passed.`

## Step 5: Review the Report

```bash
cat results/report.md
```

The report contains:
- Summary table: final KL divergence, collapse generation, curve shape for all 45 conditions
- KL divergence trajectories per generation for each agent type
- Anchor effectiveness: how much each percent of ground truth delays collapse
- Key findings summary

## How to Extend

- **Add an agent type:** Create a subclass of `BaseAgent` in `src/agents.py`, add to `AGENT_CLASSES`.
- **Add a distribution:** Add an entry to `DISTRIBUTIONS` in `src/distributions.py`.
- **Change the number of generations:** Pass `n_generations=N` to `build_configs()` in `run.py`.
- **Change the sample size:** Modify `SAMPLES_PER_GENERATION` in `src/agents.py`.
- **Add a quality metric:** Extend `_run_single()` in `src/simulation.py` to compute additional metrics per generation.
