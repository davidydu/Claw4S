"""Experiment runner: sweeps over compositions, games, sizes, and seeds.

Uses multiprocessing to parallelize independent simulations across CPU cores.

Experimental design (108 simulations total):
- 4 population compositions
- 3 game structures (symmetric, asymmetric, dominant)
- 3 population sizes (N=20, 50, 100)
- 3 random seeds per configuration
"""

from __future__ import annotations

import multiprocessing as mp
from functools import partial
from typing import Any

from src.agents import AgentType
from src.game import ALL_GAMES
from src.simulation import compute_sim_metrics


# --- Population compositions ---
# Each maps AgentType -> fraction of population; scaled to population size.

COMPOSITIONS = {
    "all_adaptive": {AgentType.ADAPTIVE: 1.0},
    "mixed_conform": {
        AgentType.CONFORMIST: 0.4,
        AgentType.ADAPTIVE: 0.4,
        AgentType.INNOVATOR: 0.1,
        AgentType.TRADITIONALIST: 0.1,
    },
    "innovator_heavy": {
        AgentType.INNOVATOR: 0.4,
        AgentType.ADAPTIVE: 0.3,
        AgentType.CONFORMIST: 0.2,
        AgentType.TRADITIONALIST: 0.1,
    },
    "traditionalist_heavy": {
        AgentType.TRADITIONALIST: 0.5,
        AgentType.CONFORMIST: 0.3,
        AgentType.ADAPTIVE: 0.1,
        AgentType.INNOVATOR: 0.1,
    },
}

POPULATION_SIZES = [20, 50, 100]
SEEDS = [42, 123, 7]
TOTAL_ROUNDS = 50_000


def _make_composition(fractions: dict[AgentType, float], n: int) -> dict[AgentType, int]:
    """Convert fractional composition to integer counts summing to n."""
    raw = {t: max(1, int(f * n)) for t, f in fractions.items()}
    # Adjust to hit exact population size
    diff = n - sum(raw.values())
    if diff != 0:
        # Add/remove from the largest group
        largest = max(raw, key=lambda t: raw[t])
        raw[largest] += diff
    return raw


def _run_single(args: tuple[str, str, int, int, int]) -> dict[str, Any]:
    """Worker function for a single simulation (pickle-friendly)."""
    comp_name, game_name, pop_size, seed, total_rounds = args
    game = ALL_GAMES[game_name]()
    composition = _make_composition(COMPOSITIONS[comp_name], pop_size)
    result = compute_sim_metrics(game, composition, total_rounds, seed)
    result["composition_name"] = comp_name
    return result


def build_experiment_grid(total_rounds: int = TOTAL_ROUNDS) -> list[tuple[str, str, int, int, int]]:
    """Build the full grid of (composition, game, size, seed, rounds)."""
    grid: list[tuple[str, str, int, int, int]] = []
    for comp_name in COMPOSITIONS:
        for game_name in ALL_GAMES:
            for pop_size in POPULATION_SIZES:
                for seed in SEEDS:
                    grid.append((comp_name, game_name, pop_size, seed, total_rounds))
    return grid


def run_experiment(total_rounds: int = TOTAL_ROUNDS, n_workers: int | None = None) -> list[dict]:
    """Run the full experiment grid using multiprocessing.

    Parameters
    ----------
    total_rounds : int
        Interactions per simulation.
    n_workers : int or None
        Number of parallel workers. Defaults to min(cpu_count, 8).

    Returns
    -------
    list[dict]
        One result dict per simulation.
    """
    grid = build_experiment_grid(total_rounds)

    if n_workers is None:
        n_workers = min(mp.cpu_count() or 4, 8)

    print(f"Running {len(grid)} simulations with {n_workers} workers...")

    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(_run_single, grid)

    print(f"Completed {len(results)} simulations.")
    return results
