"""Core cascade simulation engine.

Runs a multi-agent network simulation where one agent is shocked at a
specified round and the error propagation is tracked.

Simulation protocol:
  1. All agents start with output = 0.0 (clean state).
  2. At each round, every agent reads its neighbors' outputs from the
     previous round and computes a new output.
  3. At round T_shock, the shocked agent begins outputting a fixed
     error signal (shock_magnitude) regardless of inputs.
  4. The shock lasts for shock_duration rounds, then the agent resumes
     normal processing.
  5. An agent is "infected" if |output - clean_baseline| > ERROR_THRESHOLD.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Callable

from src.network import AdjList
from src.agents import AGENT_TYPES, NOISE_STD


# An agent is "infected" if its output deviates from baseline by this much
ERROR_THRESHOLD = 0.15


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""
    n_agents: int = 20
    topology_name: str = "ring"
    agent_type: str = "fragile"
    adj: AdjList = field(default_factory=dict)
    shock_node: int = 0
    shock_magnitude: float = 10.0
    shock_round: int = 100
    shock_duration: int = 200
    total_rounds: int = 5000
    seed: int = 42


@dataclass
class SimResult:
    """Results from a single simulation run."""
    config: SimConfig
    # Per-round outputs for each agent: outputs[round][agent_id]
    outputs: List[List[float]] = field(default_factory=list)
    # Per-round infection status: infected[round] = set of infected agent ids
    infected_per_round: List[set] = field(default_factory=list)
    # Summary metrics
    cascade_size: float = 0.0
    cascade_speed: float = float("inf")
    recovery_time: float = float("inf")
    systemic_risk: float = 0.0


def _run_single(cfg: SimConfig, apply_shock: bool) -> List[List[float]]:
    """Run one pass of the simulation, returning per-round outputs.

    When apply_shock is False, produces the clean baseline trajectory.
    Both runs use the same RNG seed so noise is identical.
    """
    rng = random.Random(cfg.seed)
    agent_fn: Callable = AGENT_TYPES[cfg.agent_type]
    n = cfg.n_agents
    adj = cfg.adj
    shock_end = cfg.shock_round + cfg.shock_duration

    prev_outputs = [0.0] * n
    curr_outputs = [0.0] * n
    all_outputs: List[List[float]] = []

    for t in range(cfg.total_rounds):
        for i in range(n):
            if apply_shock and i == cfg.shock_node and cfg.shock_round <= t < shock_end:
                # Consume the RNG draw to keep sequences aligned
                _ = rng.gauss(0, NOISE_STD)
                curr_outputs[i] = cfg.shock_magnitude
                continue

            neighbors = adj.get(i, [])
            neighbor_vals = [prev_outputs[j] for j in neighbors]
            noise = rng.gauss(0, NOISE_STD)
            curr_outputs[i] = agent_fn(neighbor_vals, noise)

        all_outputs.append(curr_outputs[:])
        prev_outputs = curr_outputs[:]

    return all_outputs


def run_simulation(cfg: SimConfig) -> SimResult:
    """Run a single cascade simulation and return results.

    Runs two identical simulations (same RNG seed):
      1. Clean baseline (no shock)
      2. Shocked simulation
    Then compares outputs round-by-round to detect infection.

    This is the main simulation loop. It intentionally avoids numpy
    to keep dependencies minimal (only stdlib).
    """
    n = cfg.n_agents

    # Run clean baseline and shocked simulation with identical noise
    baseline_outputs = _run_single(cfg, apply_shock=False)
    shocked_outputs = _run_single(cfg, apply_shock=True)

    # Storage: sample every SAMPLE_INTERVAL rounds to save memory
    SAMPLE_INTERVAL = max(1, cfg.total_rounds // 1000)
    sampled_outputs: List[List[float]] = []
    infected_per_round: List[set] = []

    ever_infected: set = set()
    infection_start: Dict[int, int] = {}
    half_infected_round: float = float("inf")
    recovery_round: float = float("inf")
    shock_end = cfg.shock_round + cfg.shock_duration

    for t in range(cfg.total_rounds):
        # Determine infected set this round
        infected_now = set()
        for i in range(n):
            deviation = abs(shocked_outputs[t][i] - baseline_outputs[t][i])
            if deviation > ERROR_THRESHOLD:
                infected_now.add(i)
                ever_infected.add(i)
                if i not in infection_start:
                    infection_start[i] = t

        # Check cascade speed: first round >= 50% infected
        if len(infected_now) >= n / 2 and half_infected_round == float("inf"):
            half_infected_round = t - cfg.shock_round

        # Check recovery: first round after shock ends with 0 infected
        if t >= shock_end and len(infected_now) == 0 and recovery_round == float("inf"):
            recovery_round = t - shock_end

        # Sample
        if t % SAMPLE_INTERVAL == 0:
            sampled_outputs.append(shocked_outputs[t])
            infected_per_round.append(infected_now)

    # Compute metrics
    cascade_size = len(ever_infected) / n
    cascade_speed = half_infected_round if half_infected_round != float("inf") else float("inf")
    recovery_time = recovery_round if recovery_round != float("inf") else float("inf")

    # Systemic risk score: composite metric
    # Higher is worse (more systemic risk)
    speed_factor = 1.0 / (1.0 + cascade_speed) if cascade_speed != float("inf") else 0.0
    recovery_factor = recovery_time / cfg.total_rounds if recovery_time != float("inf") else 1.0
    systemic_risk = cascade_size * (1.0 + speed_factor) * (1.0 + recovery_factor)

    return SimResult(
        config=cfg,
        outputs=sampled_outputs,
        infected_per_round=infected_per_round,
        cascade_size=cascade_size,
        cascade_speed=cascade_speed,
        recovery_time=recovery_time,
        systemic_risk=systemic_risk,
    )
