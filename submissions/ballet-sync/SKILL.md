---
name: ballet-sync-analysis
description: >
  Simulate emergent synchronization in ballet corps using spatially-embedded
  Kuramoto oscillators. Study phase transitions across topologies, group sizes,
  and heterogeneity levels with multi-evaluator detection.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Emergent Synchronization in Ballet Corps

This skill simulates 1,440 Kuramoto oscillator experiments to study how critical coupling strength K_c governs the spontaneous synchronization of ballet dancers. The experiment sweeps coupling strength × topology × group size × frequency heterogeneity, then detects phase transitions using sigmoid fitting, susceptibility peaks, and critical exponent analysis.

## Prerequisites

- Requires **Python 3.10+**. No internet access needed (pure simulation).
- Expected runtime: **5-10 minutes** on a single CPU.
- All commands must be run from the **submission directory** (`submissions/ballet-sync/`).

## Step 0: Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/ballet-sync/
```

All subsequent commands assume you are in this directory.

## Step 1: Environment Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

Expected: `Successfully installed numpy-... scipy-... matplotlib-... pytest-...`

Verify imports:

```bash
.venv/bin/python -c "import numpy, scipy, matplotlib; print('All imports OK')"
```

Expected: `All imports OK`

## Step 2: Run Unit Tests

Verify all simulation modules work correctly:

```bash
.venv/bin/python -m pytest tests/ -v
```

Expected: Pytest exits with `X passed` and exit code 0. All test modules cover the Kuramoto model, dancer agents, sync evaluators, experiment runner, and phase transition analysis.

## Step 3: Run the Experiment

Execute the full 1,440-simulation Kuramoto experiment:

```bash
.venv/bin/python run.py
```

Expected: Script prints progress per topology, for example:

```
[1/4] Topology: all-to-all (360 sims)...
    Done (360 sims completed)
[2/4] Topology: nearest-k (360 sims)...
    Done (360 sims completed)
...
[4/5] Generating report...
[5/5] Saving results to results/
```

Script exits with code 0. The following files are created:
- `results/results.json` — all 1,440 simulation records with evaluator scores
- `results/report.md` — markdown report with phase transition tables and key findings
- `results/statistical_tests.json` — K_c estimates, critical exponents, finite-size scaling per topology
- `results/figures/phase_transition.png`
- `results/figures/topology_comparison.png`
- `results/figures/susceptibility.png`
- `results/figures/critical_exponent.png`
- `results/figures/finite_size_scaling.png`
- `results/figures/evaluator_agreement.png`

## Step 4: Validate Results

Check that results are complete and numerically consistent:

```bash
.venv/bin/python validate.py
```

Expected output includes:
- `Records: 1440 (expected 1440)`
- `Mean kuramoto_order score at K=0: X.XXXX (expected < 0.3)` — confirms K=0 control
- `Relative difference: X.XXXX (must be < 0.01)` — dt convergence check
- `Validation passed.`

## Step 5: Review the Report

Read the generated markdown report:

```bash
cat results/report.md
```

Review the phase transition summary table (K_c per topology with 95% CI), analytical vs. empirical K_c comparison for all-to-all topology, critical exponent β table, evaluator agreement matrix, and finite-size scaling results.

## How to Extend

- **Add a topology:** Add a builder function in `src/kuramoto.py` and register it in `TOPOLOGIES`.
- **Add an evaluator:** Subclass `BaseEvaluator` in `src/evaluators.py` and add the instance to `EvaluatorPanel`.
- **Add a domain preset:** Add an entry to `DOMAIN_PRESETS` in `src/kuramoto.py` (e.g., `"orchestra": {"n": 20, "sigma": 0.4, "topology": "nearest-k"}`).
- **Change the oscillator model:** Replace the Kuramoto update rule in `KuramotoModel._deriv()` (e.g., Stuart-Landau oscillators for amplitude+phase dynamics).
- **Add spatial coupling decay:** Modify `KuramotoModel._coupling()` to weight neighbor influence by `1/distance` instead of equal weighting.
