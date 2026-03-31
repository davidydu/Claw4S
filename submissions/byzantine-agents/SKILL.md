---
name: byzantine-fault-tolerance-multi-agent
description: Simulate Byzantine fault tolerance in multi-agent voting committees. Measures how adversarial agents degrade collective decision accuracy across 3 honest voter types, 3 Byzantine strategies, 5 adversarial fractions, 3 committee sizes, and 3 seeds (405 configurations, 1000 rounds each).
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Byzantine Fault Tolerance in Multi-Agent Decision Systems

This skill runs a computational experiment studying how Byzantine (adversarial) agents degrade collective decision-making in voting committees, testing whether the classical N/3 fault tolerance bound from Lamport et al. (1982) holds for AI-like agents with different reasoning capabilities.

## Prerequisites

- Requires **Python 3.10+**. No internet access or API keys needed.
- Expected runtime: **10-20 seconds** (multiprocessing across all CPU cores).
- All commands must be run from the **submission directory** (`submissions/byzantine-agents/`).

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

Verify the analysis modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: **47 tests passed**, exit code 0.

## Step 3: Run the Experiment

Execute the full Byzantine fault tolerance experiment:

```bash
.venv/bin/python run.py
```

Expected: Script prints `[3/3] Generating report...` followed by the Markdown report, and exits with code 0. Files `results/results.json` and `results/report.md` are created.

This runs 405 simulation configurations in parallel:
- 3 honest voter types (majority, bayesian, cautious)
- 3 Byzantine strategies (random, strategic, mimicking)
- 5 Byzantine fractions (0%, 10%, 20%, 33%, 50%)
- 3 committee sizes (N=5, 9, 15)
- 3 random seeds (42, 123, 7)
- 1,000 voting rounds per configuration

## Step 4: Validate Results

Check that results were produced correctly and pass scientific sanity checks:

```bash
.venv/bin/python validate.py
```

Expected: Prints configuration counts and `Validation passed.`

## Step 5: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Expected: A Markdown report with three tables: Byzantine thresholds, amplification factors, and accuracy by honest type and fraction.

## Key Metrics

1. **Decision accuracy**: fraction of rounds where the committee selects the correct option (out of 5).
2. **Byzantine threshold (f*)**: the adversarial fraction where accuracy first drops below 50%, estimated by linear interpolation.
3. **Byzantine amplification**: ratio of accuracy degradation from strategic vs. random Byzantine agents at f=0.33 — measures how much worse coordinated adversaries are.
4. **Resilience score**: area under the accuracy-vs-fraction curve (trapezoidal rule), normalized to [0, 1].

## How to Extend

- **Add agent types**: implement the `Agent` protocol in `src/agents.py` and register in `HONEST_TYPES` or `BYZANTINE_TYPES`.
- **Change parameters**: edit `FRACTIONS`, `COMMITTEE_SIZES`, `SEEDS`, or `ROUNDS_PER_SIM` in `src/experiment.py`.
- **Different signal models**: modify `_generate_observations()` in `src/simulation.py` to change the noise structure.
- **Weighted voting**: modify the plurality counting in `run_simulation()` to support weighted votes or quorum rules.
