"""Experiment runner: sweeps over all parameter combinations.

Experimental design (324 simulations):
  - 3 initial hack discoverer counts: 1, 2, 5
  - 3 network topologies: grid, random, star
  - 3 hack detectability levels: obvious, subtle, invisible
  - 4 monitor fractions: 0.0, 0.1, 0.25, 0.5
  - 3 random seeds per combination

Uses multiprocessing to parallelize across CPU cores.
"""

from __future__ import annotations

import multiprocessing
from itertools import product
from typing import Any

import numpy as np

from src.agents import create_agent_population
from src.network import build_adjacency, neighbor_list
from src.simulation import run_simulation
from src.metrics import compute_summary_metrics


# --- Experimental parameters ---
N_AGENTS = 10
N_ROUNDS = 5000
T_DISCOVER = 50  # hack discovered at round 50

INITIAL_HACKER_COUNTS = [1, 2, 5]
TOPOLOGIES = ["grid", "random", "star"]
DETECTABILITIES = ["obvious", "subtle", "invisible"]
MONITOR_FRACTIONS = [0.0, 0.1, 0.25, 0.5]
SEEDS = [42, 123, 7]

# Total: 3 * 3 * 3 * 4 * 3 = 324 simulations


def _run_single(args: tuple) -> dict[str, Any]:
    """Run a single simulation (designed for multiprocessing.Pool.map)."""
    n_hackers, topology, detectability, monitor_frac, seed = args

    rng = np.random.default_rng(seed)

    agents = create_agent_population(N_AGENTS, monitor_frac, rng)
    adj = build_adjacency(N_AGENTS, topology, rng)
    nbrs = neighbor_list(adj)

    # Select initial hackers (first n_hackers non-monitor agents)
    non_monitor_ids = [a.agent_id for a in agents if a.agent_type != "monitor"]
    initial_hackers = non_monitor_ids[:n_hackers]

    sim_result = run_simulation(
        agents=agents,
        neighbors=nbrs,
        n_rounds=N_ROUNDS,
        initial_hackers=initial_hackers,
        t_discover=T_DISCOVER,
        detectability=detectability,
        rng=rng,
    )

    summary = compute_summary_metrics(sim_result)

    return {
        "params": {
            "n_initial_hackers": n_hackers,
            "topology": topology,
            "detectability": detectability,
            "monitor_fraction": monitor_frac,
            "seed": seed,
        },
        "metrics": summary,
    }


def build_param_grid() -> list[tuple]:
    """Build the full parameter grid as a list of tuples."""
    return list(product(
        INITIAL_HACKER_COUNTS,
        TOPOLOGIES,
        DETECTABILITIES,
        MONITOR_FRACTIONS,
        SEEDS,
    ))


def run_experiment(n_workers: int | None = None) -> list[dict[str, Any]]:
    """Run the full experiment sweep using multiprocessing.

    Parameters
    ----------
    n_workers : int or None
        Number of worker processes. Defaults to CPU count.

    Returns
    -------
    results : list of dict
        One entry per simulation with params and metrics.
    """
    param_grid = build_param_grid()
    total = len(param_grid)

    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 8)

    print(f"Running {total} simulations with {n_workers} workers...")

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_run_single, param_grid)

    print(f"Completed {len(results)} simulations.")
    return results
