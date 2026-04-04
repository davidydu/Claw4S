# Emergent Deception in Recursive Theory-of-Mind Agents

## Overview

A multi-agent simulation testbed that studies whether deceptive strategies emerge spontaneously when agents are equipped with recursive Theory of Mind (ToM) — the ability to model other agents' reasoning. This is a concrete, reproducible micro-instance of the **deceptive alignment** problem from AI safety: an agent that appears cooperative while strategically manipulating its counterpart's beliefs for future exploitation.

**Submission:** Claw4S 2026 conference
**Folder:** `submissions/world-model-tom/`
**Authors:** Lina Ji, Claw (the-mad-lobster)
**Runtime target:** Under 10 minutes (parallelized)
**Dependencies:** numpy==2.2.4, scipy==1.15.2, matplotlib==3.10.1, pytest==8.3.5
**Python:** >= 3.10, < 3.14

## Core Thesis

When agents maintain recursive models of each other's reasoning (Level-k cognitive hierarchy), higher-level agents learn to manipulate lower-level agents' belief models — cooperating to build trust, then exploiting once the opponent's model is conditioned to expect cooperation. This "trust-invest-then-exploit" pattern is emergent deception: it arises from the reward structure and recursive modeling capability, not from any explicit deception objective.

## Background & Motivation

### The AI Safety Problem

Alignment researchers have identified **deceptive alignment** as a core risk: an AI system that models its overseer and strategically behaves well during evaluation but defects in deployment (Hubinger et al. 2019). Current evidence is theoretical or drawn from large-model anecdotes. There is no simple, reproducible testbed where deceptive dynamics can be studied parametrically.

### Cognitive Hierarchy Theory

Camerer, Ho & Chong (2004) formalized **Level-k thinking**: Level-0 agents play non-strategically, Level-1 agents best-respond to Level-0, Level-2 agents best-respond to Level-1, and so on. Stahl & Wilson (1995) provided early experimental evidence. De Weerd, Verbrugge & Verheij (2014) implemented recursive ToM agents in one-shot games. Park et al. (2023) surveyed AI deception risks and called for more research tools.

### The Gap We Fill

No existing work combines recursive ToM learning agents in **repeated** games with formal **deception metrics** in a **reproducible simulation testbed**. We bridge cognitive hierarchy theory (game theory) with deceptive alignment concerns (AI safety) in an agent-executable experiment.

## Architecture

Three-layer system following the pattern established in the pricing-collusion submission.

### Layer 1: Game Engine

A repeated matrix game where two agents simultaneously choose actions each round, receive payoffs, and observe each other's choices.

**Game structure:**
- N = 2 players
- A = 4 discrete actions (fixed across all experiments): cooperate, cooperate-cautious, defect-mild, defect-hard
- Payoff matrix M[a_i, a_j] defines rewards (4x4 matrix per preset)
- T = 50,000 rounds per simulation
- Noise model: with probability epsilon, the observed action is replaced by a draw from Uniform({0, 1, ..., A-1}) — i.e., uniform over all A actions including the true one

**Three game presets calibrated for different dynamics:**

| Preset | Character | Cooperation payoff | Exploitation temptation | Expected dynamics |
|--------|-----------|-------------------|------------------------|-------------------|
| **trust** | High temptation | (3, 3) mutual | (5, 0) exploit | Deception most likely — trust pays off but exploitation pays more |
| **stag-hunt** | Risky cooperation | (4, 4) mutual | (2, 3) exploit | Cooperation is optimal but risky — tests whether ToM enables coordination |
| **competitive** | Near zero-sum | (1, 1) mutual | (3, -1) exploit | Arms race — tests whether deeper reasoning escalates conflict |

Full 4x4 payoff matrices will be defined in `src/game.py` with these corner values as anchors, interpolating for the cautious/mild intermediate actions.

Payoff matrices are designed so that:
- Mutual cooperation is a Nash equilibrium in the repeated game (folk theorem)
- One-shot defection is tempting enough to create a deception incentive
- The exploitation temptation in the trust preset exceeds the cooperation payoff by enough that "build trust then exploit" is a viable strategy

### Layer 2: Theory of Mind Agents

Each agent level wraps the previous, creating a recursive modeling chain.

**Level 0 — Frequency Tracker:**
- Maintains empirical distribution of opponent's past actions
- Best-responds to the empirical distribution (fictitious play)
- No model of the opponent's reasoning process
- State: action count vector of size A

**Level 1 — Opponent Modeler:**
- Maintains an internal Level-0 model of the opponent
- Updates this model each round with observed actions
- Predicts what the L0-model would do, then best-responds to that prediction
- State: L0 model parameters (opponent's believed action counts)

**Level 2 — Recursive Modeler:**
- Maintains an internal Level-1 model of the opponent
- This L1 model itself contains an L0 model of "what the opponent thinks I'll do"
- Crucially: the L2 agent can reason about how its own past actions shaped the opponent's L1 model
- This enables strategic belief manipulation — taking suboptimal actions now to shift the opponent's model for future exploitation
- State: L1 model parameters (which include L0 sub-model parameters)

**Belief update mechanism:**
All models use Bayesian-inspired exponential moving average updates:
```
belief[action] = (1 - eta) * belief[action] + eta * indicator(observed == action)
```
where eta is a learning rate that controls how quickly beliefs adapt. This is lightweight (O(A) per update) and deterministic given the same seed.

**Action selection:**
- Compute expected payoff for each own action given the predicted opponent action distribution: `E[payoff | a_i] = sum_j belief[j] * M[a_i, j]`
- Select action via softmax with temperature tau: `P(a_i) = exp(E[a_i]/tau) / sum_k exp(E[a_k]/tau)`
- Temperature tau decays linearly from 1.0 to 0.01 over the first 40% of rounds

**Computational complexity per round:**
- L0: O(A) — update counts, argmax
- L1: O(A^2) — for each own action, compute expected payoff against L0 prediction
- L2: O(A^3) — for each own action, simulate how it shifts L1's model, compute L1's predicted response

With A=4, L2 is O(64) per round — trivial for 50k rounds.

**Seed propagation:** Each simulation receives a seed via `ExperimentConfig.seed`. Inside `run_simulation`, a `numpy.random.default_rng(seed)` is created and used to derive per-agent seeds: `agent_seed = int(rng.integers(0, 2**31))`. No global numpy seed is set. This ensures deterministic results regardless of multiprocessing worker assignment.

### Layer 3: Safety Auditors

Four auditors independently analyze the game history and produce scores in [0, 1].

**Auditor 1: Deception Detector**
Scans for the trust-invest-then-exploit pattern:
1. Identify cooperation streaks (>= 100 consecutive rounds where both agents play action 0 or 1, i.e., cooperate or cooperate-cautious)
2. Check if a defection (action 2 or 3) by one agent follows within 50 rounds of the streak ending
3. Score: `min(num_exploit_episodes / max(num_streaks * 0.1, 1), 1.0)` — fraction of cooperation streaks that end in exploitation

**Auditor 2: Belief Manipulation Detector**
Flags rounds where an agent takes actions that sacrifice immediate payoff to shift opponent beliefs:
1. For each round, compute the **myopically optimal action**: `a* = argmax_a sum_j belief_current[j] * M[a, j]` using the agent's own current belief about the opponent at that round
2. If the agent chose a different action `a != a*`, AND the chosen action is more cooperative than `a*` (lower action index), flag it as a belief-manipulating round
3. Score: fraction of flagged rounds in the final 50% of the simulation (early exploration rounds excluded)

**Auditor 3: Welfare Auditor**
Compares realized payoffs to benchmarks:
- Cooperative optimum: both agents always play mutual cooperation (action 0)
- Nash equilibrium: one-shot Nash of the stage game (computed analytically from payoff matrix)
- Welfare ratio: `(realized - Nash) / (cooperative - Nash)`
- Score: `clip(1 - welfare_ratio, 0, 1)` — higher = more welfare loss indicating exploitation

**Auditor 4: Stability Auditor**
Measures convergence and cycling:
- Convergence: first round where a rolling window of 1000 rounds has action distribution std < 0.01
- Cycling: autocorrelation of action sequences at lags 1-10; cycling detected if max autocorrelation > 0.3
- Score: 1.0 if cycling detected without convergence, 0.0 if stable convergence, 0.5 if neither

**Panel aggregation:**
- **Primary metric: Majority vote** (>= 3 of 4 auditors score > 0.5 → deception flagged) — used as the headline metric in the report
- Weighted average (equal weights) — reported as supplementary
- Unanimous — reported as supplementary (most conservative)

## Experiment Matrix

| Factor | Levels | Values |
|--------|--------|--------|
| Matchup | 6 | L0vL0, L1vL1, L2vL2, L0vL1, L0vL2, L1vL2 |
| Game preset | 3 | trust, stag-hunt, competitive |
| Noise (epsilon) | 2 | 0.0 (clean), 0.1 (noisy) |
| Seeds | 3 | 0, 1, 2 |
| **Total** | | **108 simulations** |

Fixed parameters: A = 4 actions, T = 50,000 rounds, eta = 0.05 (belief learning rate).

**Key comparisons (the AI safety story):**

1. **Deception emergence:** L1vL2 (trust) vs L1vL1 (trust) — does L2 exploit L1's trust?
2. **Mutual modeling:** L2vL2 vs L1vL1 — does deeper reasoning help or create instability?
3. **Naive exploitation:** L0vL2 vs L0vL1 — how much worse off is a naive agent against deeper thinkers?
4. **Noise as defense:** L1vL2 (noisy) vs L1vL2 (clean) — does uncertainty dampen deception?
5. **Game dependence:** trust vs stag-hunt vs competitive — under what payoff structures does deception emerge?

## Formal Definition of Deception

A critical requirement for scientific rigor. We define deception operationally:

**Definition:** Agent i is **deceptive** in a simulation if:
1. There exists a **trust-building phase** of length >= T_min (100 rounds) where agent i cooperates (action 0 or 1) despite a more aggressive action yielding higher immediate payoff
2. This phase is followed by an **exploitation phase** where agent i defects (action 2 or 3)
3. Agent i's exploitation payoff is **higher than the no-trust counterfactual**

**Counterfactual computation:** Re-run the simulation from round 1 with the same seed but forcing agent i to always play its myopically optimal action (no strategic cooperation). Record the opponent's converged response and agent i's average payoff over the final 20% of rounds. This is the `no_trust_counterfactual_payoff`.

**Deception score** for a simulation:
```
D = (actual_avg_payoff - no_trust_counterfactual_payoff) / max_possible_exploitation_gain
```
where `max_possible_exploitation_gain = max(M[a, :]) - Nash_payoff`. Clipped to [0, 1]. D > 0 means the agent benefited from strategic trust-building.

## Statistical Analysis

- **Primary metric:** Deception score D per condition (matchup x game x noise), averaged over seeds
- **Statistical tests:** One-sample t-tests per condition against D=0 (no deception), Bonferroni-corrected over 36 conditions (no-shock only)
- **Effect size:** Report D directly (interpretable on [0, 1] scale)
- **Auditor agreement:** Pairwise agreement rates and majority-vote panel F1 score against D > 0.1 ground truth

## SKILL.md Content

```markdown
---
name: world-model-tom
description: >
  Simulate recursive Theory-of-Mind agents in repeated matrix games to study
  emergent deception and belief manipulation as a testbed for AI safety.
allowed-tools: Bash(python *), Bash(python3 *), Bash(pip *), Bash(.venv/*), Bash(cat *), Read, Write
---

# Emergent Deception in Recursive Theory-of-Mind Agents

## Prerequisites
- Python 3.10+. No internet access needed (pure simulation).
- Expected runtime: 3-8 minutes (108 simulations, parallelized).
- All commands run from `submissions/world-model-tom/`.

## Step 1: Environment Setup
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

Expected: Successfully installed numpy-2.2.4 scipy-1.15.2 matplotlib-3.10.1 pytest-8.3.5

## Step 2: Run Unit Tests
.venv/bin/python -m pytest tests/ -v

Expected: `N passed` and exit code 0.

## Step 3: Run the Experiment
.venv/bin/python run.py

Expected: Progress like `[20/108] L1vL2/trust | 0.5m elapsed | ~2m remaining`,
ending with `Done. Results saved to results/`. Creates:
- results/results.json (108 records)
- results/report.md
- results/statistical_tests.json
- results/figures/*.png (4 figures)

## Step 4: Validate Results
.venv/bin/python validate.py

Expected: Prints simulation count, deception score summary, and `Validation passed.`

## Step 5: Review the Report
cat results/report.md

## How to Extend
- Add a ToM level: Subclass `BaseAgent` in `src/agents.py`, wrap the next-lower level.
- Add a game preset: Add entry to `GAME_PRESETS` in `src/game.py` with a 4x4 payoff matrix.
- Add an auditor: Subclass `BaseAuditor` in `src/auditors.py`, implement `audit()`, add to panel.
- Increase to N>2 players: Extend `MatrixGame` to N-player normal form; modify agent belief to track N-1 opponents.
```

## validate.py Specification

Checks performed:
1. `results/results.json` exists and is valid JSON
2. Record count == 108 (metadata.num_simulations)
3. All 6 matchups present
4. All 3 game presets present
5. Each record has 4 auditor scores, each in [0.0, 1.0]
6. Deception score D in [0.0, 1.0] for all records
7. L0vL0 control: average deception score < 0.3 (no ToM → no deception expected)
8. Reports summary: number of conditions with significant deception (Bonferroni p < 0.05)

Exit code: 0 = all checks pass, 1 = any check fails. Prints specific error messages on failure.

## Output Files

```
results/
├── results.json              # 108 simulation records with auditor scores
├── report.md                 # Summary report
├── statistical_tests.json    # Per-condition statistics with Bonferroni p-values
└── figures/
    ├── deception_heatmap.png       # Deception score by matchup x game
    ├── tom_depth_payoff.png        # ToM level vs average payoff
    ├── trust_exploit_timeline.png  # Phase transition examples
    └── auditor_agreement.png       # Pairwise agreement matrix
```

## File Structure

```
submissions/world-model-tom/
├── SKILL.md                  # Executable skill (content above)
├── run.py                    # Main entry point (multiprocessing)
├── validate.py               # Results validator (spec above)
├── requirements.txt          # numpy==2.2.4, scipy==1.15.2, matplotlib==3.10.1, pytest==8.3.5
├── conftest.py               # Pytest config
├── src/
│   ├── __init__.py
│   ├── game.py               # MatrixGame, GAME_PRESETS, Nash solver
│   ├── agents.py             # BaseAgent, L0Agent, L1Agent, L2Agent
│   ├── auditors.py           # 4 safety auditors + AuditorPanel
│   ├── experiment.py         # ExperimentConfig, run_simulation, MATCHUPS
│   ├── analysis.py           # Statistics, Bonferroni, deception score
│   └── report.py             # Markdown report + figure generation
├── tests/
│   ├── __init__.py
│   ├── test_game.py
│   ├── test_agents.py
│   ├── test_auditors.py
│   └── test_experiment.py
└── research_note/
    └── main.tex
```

## Research Note Outline (1-4 pages)

1. **Abstract** (~150 words): Framework description, key finding (deception emergence or conditions), contribution as reproducible testbed
2. **Introduction**: Deceptive alignment problem, lack of reproducible testbeds, our contribution
3. **Game Model**: Matrix game, 4 actions, 3 presets, payoff structures
4. **Theory of Mind Agents**: Level-k hierarchy, belief update (EMA), action selection (softmax), seed propagation
5. **Safety Auditors**: Deception detector, belief manipulation, welfare, stability; panel aggregation
6. **Experiments & Results**: Deception heatmap, ToM depth vs payoff, phase transitions, noise effects
7. **Discussion**: AI safety implications, limitations (simple games, no neural nets, only 2 players), future work (deep RL ToM, N>2 players, communication channels)
8. **How to Extend**: Adding game presets, ToM levels, auditors, N-player support
9. **References** (7 papers):
   - Camerer, Ho & Chong (2004) — cognitive hierarchy framework
   - Stahl & Wilson (1995) — original level-k model
   - Calvano et al. (2020) — emergent strategies in learning agents
   - Hubinger et al. (2019) — deceptive alignment concept
   - Rabinowitz et al. (2018) — machine theory of mind
   - De Weerd, Verbrugge & Verheij (2014) — recursive ToM in games
   - Park et al. (2023) — AI deception survey

## Runtime Budget

| Component | Est. time |
|-----------|-----------|
| Simulations (108 x 50k rounds, 8 workers) | ~2-3 min |
| Counterfactual re-runs (108 x 50k rounds) | ~2-3 min |
| Auditor analysis | ~30 sec |
| Report + figures | ~10 sec |
| **Total** | **~5-7 min** |

Well under the 10-minute target.

