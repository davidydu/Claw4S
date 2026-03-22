"""Core simulation engine for information cascades.

Runs a sequence of agents who each:
1. Receive a private signal about the true binary state.
2. Observe all predecessors' actions.
3. Choose action A (0) or B (1).

A cascade is detected when an agent's action differs from its private signal
(i.e., the agent ignores its signal and follows the crowd).
"""

import random

from src.agents import ACTION_A, ACTION_B, make_agent


def generate_signals(
    true_state: int,
    n_agents: int,
    signal_quality: float,
    rng: random.Random,
) -> list[int]:
    """Generate private signals for all agents.

    Each signal matches the true state with probability signal_quality.
    """
    signals = []
    for _ in range(n_agents):
        if rng.random() < signal_quality:
            signals.append(true_state)
        else:
            signals.append(1 - true_state)
    return signals


def detect_cascade(actions: list[int], signals: list[int]) -> dict:
    """Analyze a completed sequence for cascade properties.

    A cascade starts at position i if agent i's action differs from its signal
    AND at least 2 consecutive preceding agents chose the same action.

    Returns dict with:
        - cascade_formed: bool
        - cascade_start: int or None (first position where cascade detected)
        - cascade_action: int or None (the action being cascaded on)
        - cascade_length: int (consecutive agents in cascade from start)
        - agents_in_cascade: int (total agents whose action != signal after start)
    """
    n = len(actions)
    if n < 3:
        return {
            "cascade_formed": False,
            "cascade_start": None,
            "cascade_action": None,
            "cascade_length": 0,
            "agents_in_cascade": 0,
        }

    cascade_start = None
    cascade_action = None

    for i in range(2, n):
        # Check if at least 2 preceding agents chose the same action
        if actions[i - 1] == actions[i - 2]:
            dominant = actions[i - 1]
            # Check if agent i ignores its signal to follow the crowd
            if actions[i] == dominant and signals[i] != dominant:
                cascade_start = i
                cascade_action = dominant
                break

    if cascade_start is None:
        return {
            "cascade_formed": False,
            "cascade_start": None,
            "cascade_action": None,
            "cascade_length": 0,
            "agents_in_cascade": 0,
        }

    # Measure cascade length: consecutive agents matching cascade_action from start
    cascade_length = 0
    for i in range(cascade_start, n):
        if actions[i] == cascade_action:
            cascade_length += 1
        else:
            break

    # Count agents who ignored signal
    agents_in_cascade = 0
    for i in range(cascade_start, n):
        if actions[i] != signals[i]:
            agents_in_cascade += 1

    return {
        "cascade_formed": True,
        "cascade_start": cascade_start,
        "cascade_action": cascade_action,
        "cascade_length": cascade_length,
        "agents_in_cascade": agents_in_cascade,
    }


def run_single_simulation(
    agent_type: str,
    n_agents: int,
    signal_quality: float,
    true_state: int,
    seed: int,
) -> dict:
    """Run a single cascade simulation.

    Args:
        agent_type: One of "bayesian", "heuristic", "contrarian", "noisy_bayesian".
        n_agents: Number of agents in the sequence.
        signal_quality: Probability that each signal matches true state.
        true_state: The true binary state (0=A or 1=B).
        seed: Random seed for reproducibility.

    Returns:
        Dict with simulation parameters and results.
    """
    rng = random.Random(seed)

    signals = generate_signals(true_state, n_agents, signal_quality, rng)
    agent = make_agent(agent_type, rng=rng)

    actions: list[int] = []
    for i in range(n_agents):
        action = agent.choose(signals[i], actions.copy(), signal_quality)
        actions.append(action)

    cascade_info = detect_cascade(actions, signals)

    # Compute final majority action
    count_a = actions.count(ACTION_A)
    count_b = actions.count(ACTION_B)
    majority_action = ACTION_A if count_a >= count_b else ACTION_B
    majority_correct = majority_action == true_state

    return {
        "agent_type": agent_type,
        "n_agents": n_agents,
        "signal_quality": signal_quality,
        "true_state": true_state,
        "seed": seed,
        "actions": actions,
        "signals": signals,
        "cascade_formed": cascade_info["cascade_formed"],
        "cascade_start": cascade_info["cascade_start"],
        "cascade_action": cascade_info["cascade_action"],
        "cascade_length": cascade_info["cascade_length"],
        "agents_in_cascade": cascade_info["agents_in_cascade"],
        "cascade_correct": (
            cascade_info["cascade_action"] == true_state
            if cascade_info["cascade_formed"]
            else None
        ),
        "majority_correct": majority_correct,
    }
