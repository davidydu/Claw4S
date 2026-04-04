"""Experiment runner: generates all simulation configurations and runs them
in parallel using multiprocessing.

Experiment design (324 simulations):
  6 topologies x 3 agent types x 3 shock magnitudes x 2 shock locations x 3 seeds

Each simulation: N=20 agents, 5000 rounds.
"""

from __future__ import annotations

import multiprocessing
import os
import random
import time
from typing import List, Dict, Any, Mapping, Sequence

from src.network import build_network, max_degree_nodes, TOPOLOGIES
from src.agents import AGENT_TYPES
from src.simulation import SimConfig, run_simulation

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


def select_shock_node(adj: Dict[int, List[int]], shock_location: str, seed: int) -> int:
    """Pick a reproducible shock node for the requested location policy.

    - hub: first max-degree node (deterministic)
    - random: uniformly sampled among non-hub nodes when available, otherwise
      sampled among all nodes (regular graphs with no strict non-hubs)
    """
    hub_nodes = max_degree_nodes(adj)
    if shock_location == "hub":
        return hub_nodes[0]
    if shock_location == "random":
        hub_set = set(hub_nodes)
        non_hub_nodes = [node for node in sorted(adj) if node not in hub_set]
        candidates = non_hub_nodes if non_hub_nodes else sorted(adj)
        return random.Random(seed).choice(candidates)
    raise ValueError(f"Unknown shock_location: {shock_location}")


def _build_configs(
    topology_names: Sequence[str] | None = None,
    agent_type_names: Sequence[str] | None = None,
    shock_magnitudes: Mapping[str, float] | None = None,
    shock_locations: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> List[SimConfig]:
    """Generate simulation configurations for the provided experiment axes."""
    topology_axis = list(topology_names) if topology_names is not None else list(TOPOLOGY_NAMES)
    agent_axis = list(agent_type_names) if agent_type_names is not None else list(AGENT_TYPE_NAMES)
    magnitude_axis = dict(shock_magnitudes) if shock_magnitudes is not None else dict(SHOCK_MAGNITUDES)
    location_axis = list(shock_locations) if shock_locations is not None else list(SHOCK_LOCATIONS)
    seed_axis = list(seeds) if seeds is not None else list(SEEDS)

    configs = []
    for topo_name in topology_axis:
        for agent_type in agent_axis:
            for _, mag_val in magnitude_axis.items():
                for shock_loc in location_axis:
                    for seed in seed_axis:
                        adj = build_network(topo_name, N_AGENTS, seed=seed)
                        shock_node = select_shock_node(adj, shock_loc, seed)

                        cfg = SimConfig(
                            n_agents=N_AGENTS,
                            topology_name=topo_name,
                            agent_type=agent_type,
                            adj=adj,
                            shock_node=shock_node,
                            shock_location=shock_loc,
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
    hub_nodes = set(max_degree_nodes(cfg.adj))
    return {
        "topology": cfg.topology_name,
        "agent_type": cfg.agent_type,
        "shock_magnitude": cfg.shock_magnitude,
        "shock_location": cfg.shock_location,
        "shock_node": cfg.shock_node,
        "shock_node_is_hub": cfg.shock_node in hub_nodes,
        "has_non_hub_nodes": len(hub_nodes) < len(cfg.adj),
        "seed": cfg.seed,
        "cascade_size": result.cascade_size,
        "cascade_speed": result.cascade_speed,
        "recovery_time": result.recovery_time,
        "systemic_risk": result.systemic_risk,
    }


def run_experiment(
    n_workers: int | None = None,
    topology_names: Sequence[str] | None = None,
    agent_type_names: Sequence[str] | None = None,
    shock_magnitudes: Mapping[str, float] | None = None,
    shock_locations: Sequence[str] | None = None,
    seeds: Sequence[int] | None = None,
) -> List[Dict[str, Any]]:
    """Run all 324 simulations in parallel and return results.

    Args:
        n_workers: Number of parallel workers. Defaults to CPU count.
        topology_names: Optional subset of topologies.
        agent_type_names: Optional subset of agent types.
        shock_magnitudes: Optional magnitude-label mapping.
        shock_locations: Optional subset of shock locations.
        seeds: Optional subset of random seeds.

    Returns:
        List of result dicts, one per simulation.
    """
    if n_workers is None:
        n_workers = min(os.cpu_count() or 4, 8)

    topology_axis = list(topology_names) if topology_names is not None else list(TOPOLOGY_NAMES)
    agent_axis = list(agent_type_names) if agent_type_names is not None else list(AGENT_TYPE_NAMES)
    magnitude_axis = dict(shock_magnitudes) if shock_magnitudes is not None else dict(SHOCK_MAGNITUDES)
    location_axis = list(shock_locations) if shock_locations is not None else list(SHOCK_LOCATIONS)
    seed_axis = list(seeds) if seeds is not None else list(SEEDS)

    configs = _build_configs(
        topology_names=topology_axis,
        agent_type_names=agent_axis,
        shock_magnitudes=magnitude_axis,
        shock_locations=location_axis,
        seeds=seed_axis,
    )
    expected = len(topology_axis) * len(agent_axis) * len(magnitude_axis) * len(location_axis) * len(seed_axis)
    assert len(configs) == expected, f"Expected {expected} configs, got {len(configs)}"

    print(f"Running {len(configs)} simulations with {n_workers} workers...")
    t0 = time.time()

    if n_workers <= 1:
        results = [_run_one(cfg) for cfg in configs]
    else:
        with multiprocessing.Pool(processes=n_workers) as pool:
            results = pool.map(_run_one, configs)

    elapsed = time.time() - t0
    print(f"Completed {len(results)} simulations in {elapsed:.1f}s")

    return results
