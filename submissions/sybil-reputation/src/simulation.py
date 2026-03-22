"""Core simulation loop for the Sybil reputation experiment.

Runs a single simulation: N honest agents + K Sybil agents interact
for a fixed number of rounds. Each round, random pairs trade and rate
each other. Sybil agents inject fake ratings according to their strategy.
At the end, reputation is computed and metrics evaluated.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Tuple

from .agents import Agent, create_honest_agents, create_sybil_agents
from .reputation import Rating, ALGORITHMS
from .sybil_strategies import STRATEGIES
from .metrics import compute_all_metrics


def _honest_transaction(
    a1: Agent, a2: Agent, rng, round_num: int
) -> List[Rating]:
    """Simulate an honest transaction between two agents.

    Each agent rates the other based on the other's true quality
    plus Gaussian noise (sigma=0.1), clipped to [0, 1].
    """
    noise1 = float(rng.normal(0, 0.1))
    noise2 = float(rng.normal(0, 0.1))
    r1 = float(np.clip(a2.true_quality + noise1, 0, 1))
    r2 = float(np.clip(a1.true_quality + noise2, 0, 1))
    return [
        (a1.agent_id, a2.agent_id, r1, round_num),
        (a2.agent_id, a1.agent_id, r2, round_num),
    ]


def run_single_sim(
    n_honest: int,
    n_sybil: int,
    algorithm_name: str,
    strategy_name: str,
    n_rounds: int,
    seed: int,
    trades_per_round: int = 5,
) -> Dict:
    """Run one simulation and return results.

    Args:
        n_honest: Number of honest agents.
        n_sybil: Number of Sybil agents (K).
        algorithm_name: Key into ALGORITHMS dict.
        strategy_name: Key into STRATEGIES dict (ignored if n_sybil == 0).
        n_rounds: Number of simulation rounds.
        seed: Random seed for reproducibility.
        trades_per_round: Number of honest-honest trades per round.

    Returns:
        Dict with config, metrics, and per-agent reputation scores.
    """
    rng = np.random.default_rng(seed)

    honest_agents = create_honest_agents(n_honest, rng)
    sybil_agents = (
        create_sybil_agents(n_sybil, start_id=n_honest, controller_id=9999)
        if n_sybil > 0
        else []
    )
    all_agents = honest_agents + sybil_agents

    strategy_fn = STRATEGIES.get(strategy_name) if n_sybil > 0 else None
    ledger: List[Rating] = []

    # Sybil agents join at 10% of total rounds (they are newcomers)
    sybil_join_round = n_rounds // 10

    for rnd in range(n_rounds):
        # Honest agents trade with each other
        if len(honest_agents) >= 2:
            for _ in range(trades_per_round):
                i, j = rng.choice(len(honest_agents), size=2, replace=False)
                txn = _honest_transaction(
                    honest_agents[i], honest_agents[j], rng, rnd
                )
                ledger.extend(txn)

        # Sybil agents inject fake ratings only after they join
        if strategy_fn is not None and n_sybil > 0 and rnd >= sybil_join_round:
            fake_ratings = strategy_fn(sybil_agents, honest_agents, rng)
            for rater, ratee, value in fake_ratings:
                ledger.append((rater, ratee, value, rnd))

        # Age honest agents every round; Sybils only after joining
        for a in honest_agents:
            a.account_age += 1
        if rnd >= sybil_join_round:
            for s in sybil_agents:
                s.account_age += 1

        # Whitewashing: reset Sybil ages periodically after joining
        if (
            strategy_name == "whitewashing"
            and rnd > sybil_join_round
            and (rnd - sybil_join_round) % 500 == 0
        ):
            for s in sybil_agents:
                s.account_age = 0

    # Compute reputation scores
    algo_fn = ALGORITHMS[algorithm_name]
    if algorithm_name == "weighted_history":
        scores = algo_fn(all_agents, ledger, n_rounds)
    else:
        scores = algo_fn(all_agents, ledger)

    # Update agent objects
    for a in all_agents:
        a.reputation_score = scores.get(a.agent_id, 0.5)

    # Compute metrics
    metrics = compute_all_metrics(honest_agents, sybil_agents, scores)

    return {
        "config": {
            "n_honest": n_honest,
            "n_sybil": n_sybil,
            "algorithm": algorithm_name,
            "strategy": strategy_name,
            "n_rounds": n_rounds,
            "seed": seed,
        },
        "metrics": metrics,
        "honest_reputations": {
            a.agent_id: {
                "true_quality": a.true_quality,
                "reputation": scores.get(a.agent_id, 0.5),
            }
            for a in honest_agents
        },
        "sybil_reputations": {
            a.agent_id: scores.get(a.agent_id, 0.5) for a in sybil_agents
        },
    }
