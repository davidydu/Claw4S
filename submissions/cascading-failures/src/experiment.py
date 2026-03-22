"""Experiment runner: generates all simulation configurations and runs them
in parallel using multiprocessing.

Experiment design (324 simulations):
  6 topologies x 3 agent types x 3 shock magnitudes x 2 shock locations x 3 seeds

Each simulation: N=20 agents, 5000 rounds.
"""

from __future__ import annotations

import multiprocessing
import os
import time
from dataclasses import asdict
from typing import List, Dict, Any

from src.network import build_network, hub_node, TOPOLOGIES
from src.agents import AGENT_TYPES
from src.simulation import SimConfig, SimResult, run_simulation

# Experiment parameters
N_AGENTS = 20
TOTAL_ROUNDS = 5000
SHOCK_ROUND = 100
SHOCK_DURATION = 200

TOPOLOGY_NAMES = list(TOPOLOGIES.keys())
AGENT_TYPE_NAMES = list(AGENT_TYPES.keys())
SHOCK_MAGNITUDES = {"mild": 2.0, "moderate": 10.0, "severe": 50.0}
SHOCK_LOCATIONS = ["random", "hub"]
SEEDS = [42, 123, 777]


def _build_configs() -> List[SimConfig]:
    """Generate all 324 simulation configurations."""
    configs = []
    for topo_name in TOPOLOGY_NAMES:
        for agent_type in AGENT_TYPE_NAMES:
            for mag_label, mag_val in SHOCK_MAGNITUDES.items():
                for shock_loc in SHOCK_LOCATIONS:
                    for seed in SEEDS:
                        adj = build_network(topo_name, N_AGENTS, seed=seed)
                        if shock_loc == "hub":
                            shock_node = hub_node(adj)
                        else:
                            # "random" = node farthest from hub
                            import random as _rng
                            r = _rng.Random(seed)
                            hub = hub_node(adj)
                            non_hub = [i for i in range(N_AGENTS) if i != hub]
                            shock_node = r.choice(non_hub) if non_hub else 0

                        cfg = SimConfig(
                            n_agents=N_AGENTS,
                            topology_name=topo_name,
                            agent_type=agent_type,
                            adj=adj,
                            shock_node=shock_node,
                            shock_magnitude=mag_val,
                            shock_round=SHOCK_ROUND,
                            shock_duration=SHOCK_DURATION,
                            total_rounds=TOTAL_ROUNDS,
                            seed=seed,
                        )
                        configs.append(cfg)
    return configs


def _run_one(cfg: SimConfig) -> Dict[str, Any]:
    """Run a single simulation and return serializable result dict."""
    result = run_simulation(cfg)
    return {
        "topology": cfg.topology_name,
        "agent_type": cfg.agent_type,
        "shock_magnitude": cfg.shock_magnitude,
        "shock_location": "hub" if cfg.shock_node == hub_node(cfg.adj) else "random",
        "shock_node": cfg.shock_node,
        "seed": cfg.seed,
        "cascade_size": result.cascade_size,
        "cascade_speed": result.cascade_speed,
        "recovery_time": result.recovery_time,
        "systemic_risk": result.systemic_risk,
    }


def run_experiment(n_workers: int | None = None) -> List[Dict[str, Any]]:
    """Run all 324 simulations in parallel and return results.

    Args:
        n_workers: Number of parallel workers. Defaults to CPU count.

    Returns:
        List of result dicts, one per simulation.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    configs = _build_configs()
    expected = len(TOPOLOGY_NAMES) * len(AGENT_TYPE_NAMES) * len(SHOCK_MAGNITUDES) * len(SHOCK_LOCATIONS) * len(SEEDS)
    assert len(configs) == expected, f"Expected {expected} configs, got {len(configs)}"

    print(f"Running {len(configs)} simulations with {n_workers} workers...")
    t0 = time.time()

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_run_one, configs)

    elapsed = time.time() - t0
    print(f"Completed {len(results)} simulations in {elapsed:.1f}s")

    return results
