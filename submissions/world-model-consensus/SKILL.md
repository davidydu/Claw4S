# Skill: World Model Consensus in Multi-Agent Coordination

## Goal
Investigate whether there exists a sharp **consensus threshold** — a critical
level of prior disagreement — beyond which multi-agent coordination collapses.
Run 396 agent-based simulations, measure coordination rates, detect phase
transitions, and generate a reproducible analysis report.

## Prerequisites
- Python 3.11+
- No GPU, API keys, or network access required
- All computation is local (agent-based simulation)

## Steps

### Step 0 — Get the Code

Clone the repository and navigate to the submission directory:

```bash
git clone https://github.com/davidydu/Claw4S.git
cd Claw4S/submissions/world-model-consensus/
```

All subsequent commands assume you are in this directory.

### Step 1 — Create virtual environment and install dependencies
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```
**Expected output:** Clean install of numpy==2.2.3, scipy==1.15.2, matplotlib==3.10.1, pytest==8.3.5.

### Step 2 — Run unit tests
```bash
.venv/bin/python -m pytest tests/ -v
```
**Expected output:** 51 tests passed, 0 failed.

### Step 3 — Run the experiment
```bash
.venv/bin/python run.py
```
**Expected output:** 396 simulations complete. Prints phase transition table
and coordination rate matrix. Generates 4 figures and a Markdown report in
`results/`.

Runtime: ~10 seconds on a 12-core machine.

### Step 4 — Validate results
```bash
.venv/bin/python validate.py
```
**Expected output:** 27/27 validation checks passed.

## Output Files
| File | Description |
|------|-------------|
| `results/raw_results.json` | Per-simulation metrics (396 entries) |
| `results/summary_table.json` | Aggregated metrics per condition |
| `results/phase_transitions.json` | Detected transition points and sharpness |
| `results/report.md` | Full Markdown analysis report |
| `results/fig1_coordination_vs_disagreement.png` | Main result: coordination rate vs disagreement |
| `results/fig2_consensus_time.png` | Consensus speed vs disagreement |
| `results/fig3_group_size_effect.png` | Group size scaling (N=3,4,6) |
| `results/fig4_fairness.png` | Majority-preference fraction |

## Key Findings
1. A sharp phase transition at d~0.51 for stubborn and mixed compositions
   (sharpness ~13, coordination drops from 1.0 to 0.0 in one step).
2. Adaptive agents with epsilon-greedy exploration (5%) maintain ~85%
   coordination at ALL disagreement levels — exploration breaks symmetry.
3. Leader-follower groups partially bridge the gap: 2 of 3 seeds maintain
   coordination even at maximal disagreement.
4. Coordination rate at d=0 scales with group size: N=3 (88.5%), N=4 (85.1%),
   N=6 (78.9%), bounded by epsilon noise (0.95^N).

## Experiment Design
- **Game:** Pure coordination (payoff 1 if all choose same action, 0 otherwise)
- **Agents:** 4 types — Stubborn, Adaptive (EMA + epsilon-greedy), Leader, Follower
- **Matrix:** 4 compositions x 11 disagreement levels x 3 group sizes x 3 seeds
- **Rounds:** 10,000 per simulation
- **Metrics:** Coordination rate (final 20%), consensus time, welfare, fairness

## How to Extend
- Change `DISAGREEMENT_LEVELS` in `src/experiment.py` for finer resolution
- Add new agent types in `src/agents.py` (subclass `BaseAgent`)
- Modify `COMPOSITIONS` in `src/experiment.py` for new group structures
- Adjust `epsilon` and `learning_rate` parameters to study exploration-exploitation
- Increase `n_rounds` in `SimulationConfig` for longer convergence studies
