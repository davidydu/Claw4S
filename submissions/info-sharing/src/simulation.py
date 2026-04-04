"""Simulation runner: executes one information-sharing game."""

from __future__ import annotations

import numpy as np

from src.agents import Agent, create_agents
from src.environment import EnvConfig, Environment
from src.metrics import compute_round_metrics, compute_summary_metrics


def run_simulation(
    composition: list[str],
    competition: float,
    complementarity: float,
    n_rounds: int,
    seed: int,
) -> dict:
    """Run a single information-sharing simulation.

    Parameters
    ----------
    composition : list[str]
        Agent types, e.g. ["open", "open", "open", "open"].
    competition : float
        Competition level in [0, 1].
    complementarity : float
        Info complementarity in [0, 1].
    n_rounds : int
        Number of rounds to simulate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with config, summary metrics, and time-series data.
    """
    rng = np.random.default_rng(seed)
    config = EnvConfig(
        n_agents=len(composition),
        competition=competition,
        complementarity=complementarity,
    )
    env = Environment(config, rng)
    agents = create_agents(composition, rng)
    agent_types = [a.type_name() for a in agents]

    # History tracking
    round_metrics_list: list[dict] = []
    per_agent_sharing: list[np.ndarray] = []
    per_agent_payoffs: list[np.ndarray] = []

    # Per-agent rolling history for agents that need it
    history_payoffs: list[list[float]] = [[] for _ in agents]
    history_others_sharing: list[list[float]] = [[] for _ in agents]

    for t in range(n_rounds):
        # Agents choose disclosure
        disclosures = np.array([
            agents[i].choose_disclosure(
                t, history_payoffs[i], history_others_sharing[i]
            )
            for i in range(len(agents))
        ])

        # Environment step
        payoffs, sharing, errors, group_welfare = env.step(disclosures)

        # Update histories
        for i in range(len(agents)):
            history_payoffs[i].append(float(payoffs[i]))
            others_avg = float(np.mean(
                [disclosures[j] for j in range(len(agents)) if j != i]
            ))
            history_others_sharing[i].append(others_avg)

        # Compute and store metrics
        rm = compute_round_metrics(payoffs, sharing, errors)
        round_metrics_list.append(rm)
        per_agent_sharing.append(sharing.copy())
        per_agent_payoffs.append(payoffs.copy())

    # Compute summary
    summary = compute_summary_metrics(
        round_metrics_list, agent_types, per_agent_sharing, per_agent_payoffs, n_rounds
    )

    # Subsample time series for storage (every 100th round)
    step = max(1, n_rounds // 100)
    ts_indices = list(range(0, n_rounds, step))
    time_series = {
        "rounds": ts_indices,
        "mean_sharing": [round_metrics_list[i]["mean_sharing"] for i in ts_indices],
        "group_welfare": [round_metrics_list[i]["group_welfare"] for i in ts_indices],
        "welfare_gap": [round_metrics_list[i]["welfare_gap"] for i in ts_indices],
        "info_asymmetry": [round_metrics_list[i]["info_asymmetry"] for i in ts_indices],
    }

    return {
        "config": {
            "composition": composition,
            "agent_types": agent_types,
            "competition": competition,
            "complementarity": complementarity,
            "n_rounds": n_rounds,
            "seed": seed,
        },
        "summary": summary,
        "time_series": time_series,
    }
