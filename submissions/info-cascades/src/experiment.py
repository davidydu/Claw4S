"""Experiment runner for information cascade simulations.

Runs the full factorial experiment:
  4 agent types x 3 signal qualities x 3 sequence lengths x 2 true states x 3 seeds
  = 216 simulations

Uses multiprocessing for parallel execution.
"""

from __future__ import annotations
import multiprocessing
from itertools import product

from src.simulation import run_single_simulation
from src.metrics import compute_group_metrics, proportion_ci_95

# Experiment configuration
AGENT_TYPES = ["bayesian", "heuristic", "contrarian", "noisy_bayesian"]
SIGNAL_QUALITIES = [0.6, 0.7, 0.9]
SEQUENCE_LENGTHS = [10, 20, 50]
TRUE_STATES = [0, 1]  # A=0, B=1 — symmetric by design
SEEDS = [42, 123, 7]

AGENT_LABELS = {
    "bayesian": "Bayesian",
    "heuristic": "Heuristic",
    "contrarian": "Contrarian",
    "noisy_bayesian": "Noisy-Bayesian",
}


def _run_task(args: tuple) -> dict:
    """Wrapper for multiprocessing — unpacks args tuple."""
    agent_type, n_agents, signal_quality, true_state, seed = args
    return run_single_simulation(agent_type, n_agents, signal_quality, true_state, seed)


def build_task_list() -> list[tuple]:
    """Generate all (agent_type, n_agents, signal_quality, true_state, seed) tasks."""
    tasks = []
    for agent_type, sq, n, ts, seed in product(
        AGENT_TYPES, SIGNAL_QUALITIES, SEQUENCE_LENGTHS, TRUE_STATES, SEEDS
    ):
        tasks.append((agent_type, n, sq, ts, seed))
    return tasks


def run_experiment(n_workers: int | None = None) -> list[dict]:
    """Run full experiment with multiprocessing.

    Args:
        n_workers: Number of worker processes. Defaults to CPU count.

    Returns:
        List of simulation result dicts.
    """
    tasks = build_task_list()
    if n_workers is None:
        n_workers = min(multiprocessing.cpu_count(), 8)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(_run_task, tasks)
    return results


def group_results(
    results: list[dict],
) -> dict[tuple[str, float, int], list[dict]]:
    """Group results by (agent_type, signal_quality, n_agents).

    Aggregates across true_state and seed, which are replication dimensions.
    """
    groups: dict[tuple[str, float, int], list[dict]] = {}
    for r in results:
        key = (r["agent_type"], r["signal_quality"], r["n_agents"])
        groups.setdefault(key, []).append(r)
    return groups


def compute_all_metrics(results: list[dict]) -> list[dict]:
    """Compute metrics for each (agent_type, signal_quality, n_agents) group.

    Returns list of dicts, each containing group key and metric values.
    """
    groups = group_results(results)
    output = []
    for (agent_type, sq, n), group in sorted(groups.items()):
        metrics = compute_group_metrics(group)
        n_cascades = sum(1 for r in group if r["cascade_formed"])
        n_correct = sum(1 for r in group if r["cascade_formed"] and r["cascade_correct"])
        form_lo, form_hi = proportion_ci_95(n_cascades, len(group))
        acc_lo, acc_hi = proportion_ci_95(n_correct, n_cascades) if n_cascades > 0 else (None, None)
        output.append({
            "agent_type": agent_type,
            "agent_label": AGENT_LABELS[agent_type],
            "signal_quality": sq,
            "n_agents": n,
            "n_simulations": metrics["n_simulations"],
            "cascade_formation_rate": metrics["cascade_formation_rate"],
            "cascade_formation_ci": (form_lo, form_hi),
            "cascade_accuracy": metrics["cascade_accuracy"],
            "cascade_accuracy_ci": (acc_lo, acc_hi),
            "cascade_fragility": metrics["cascade_fragility"],
            "mean_cascade_length": metrics["mean_cascade_length"],
            "majority_accuracy": metrics["majority_accuracy"],
        })
    return output
