---
name: sybil-reputation
description: Simulate Sybil attacks on multi-agent reputation networks. Tests 4 reputation algorithms (simple average, weighted-by-history, PageRank trust, EigenTrust) against 3 Sybil strategies (ballot stuffing, bad-mouthing, whitewashing) across 5 attacker counts. Measures reputation accuracy, Sybil detection, honest welfare, and market efficiency.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Sybil Resilience in AI Agent Reputation Networks

This skill simulates Sybil attacks on multi-agent reputation systems and measures which reputation algorithms are most resilient. It runs 156 simulations across a full parameter grid with multiprocessing.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure simulation).
- Expected runtime: **2-4 minutes** on a modern machine (12 cores).
- All commands must be run from the **submission directory** (`submissions/sybil-reputation/`).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/sybil-reputation/
```

All subsequent commands assume you are in this directory.

## Step 1: Environment Setup

Create a virtual environment and install requirements:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Verify the local modules are importable:

```bash
.venv/bin/python -c "from src.simulation import run_single_sim; import validate, pytest; print('Environment OK')"
```

Expected output: `Environment OK`

## Step 2: Run Unit Tests

Verify the simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

(`pytest` is provided as a local module in this submission for offline execution.)

Expected: `31 passed` and exit code 0.

## Step 3: Run Diagnostic

Sanity-check with a small simulation grid before the full experiment:

```bash
.venv/bin/python run.py --diagnostic
```

Expected: Prints 4 diagnostic result rows (algorithm, K value, and four metrics) and exits with code 0.

## Step 4: Run the Full Experiment

Execute the 156-simulation grid (4 algorithms x 3 strategies x 5 Sybil counts x 3 seeds, with K=0 baselines):

```bash
.venv/bin/python run.py
```

Expected: Script prints `[3/3] Saved results to results/results.json` and generates `results/report.md`. Runtime ~2-4 minutes.

This runs:
1. 20 honest agents with true quality in [0.2, 0.9]
2. Sybil agents (K=0,2,5,10,20) join at round 500 of 5000
3. Honest agents trade and rate each other; Sybils inject fake ratings
4. Reputation computed via each algorithm after all rounds
5. Four metrics evaluated: reputation accuracy, Sybil detection rate, honest welfare, market efficiency

## Step 5: Validate Results

Check that results are complete and scientifically sound:

```bash
.venv/bin/python validate.py
```

Expected: `Validation passed.` with 156 simulations, baseline accuracy > 0.5.

## Step 6: Review the Report

Read the generated report:

```bash
cat results/report.md
```

Expected: Four tables (accuracy, detection, welfare, efficiency) plus key findings. In typical runs, PageRank remains the top performer at high Sybil counts, while simple average degrades notably under attack.

## How to Extend

- **Add algorithms:** Implement a new function in `src/reputation.py` matching the signature `(agents, ledger) -> Dict[int, float]` and register it in the `ALGORITHMS` dict.
- **Add strategies:** Implement in `src/sybil_strategies.py` matching `(sybil_agents, honest_agents, rng) -> List[Tuple[int, int, float]]` and register in `STRATEGIES`.
- **Change parameters:** Edit `src/experiment.py` constants: `N_HONEST`, `SYBIL_COUNTS`, `SEEDS`, `N_ROUNDS`.
- **Scale up:** Increase `N_ROUNDS` for more statistical power, or add more seeds for tighter confidence intervals.
