"""Core simulation engine for reward-hacking propagation.

Each simulation round:
  1. All agents sample rewards based on current strategy.
  2. Each non-monitor agent observes neighbors' proxy rewards.
  3. If any neighbor has higher proxy reward, the agent may adopt that
     neighbor's strategy with probability sigmoid(beta * reward_gap).
  4. Monitors compute proxy-true divergence across observed neighbors.
  5. If a monitor detects divergence above threshold, it "quarantines"
     adjacent hacking agents (resets them to honest).
"""

from __future__ import annotations

import numpy as np

from src.agents import Agent, sample_reward, HACK_PROXY_MEAN, HONEST_PROXY_MEAN


# Detectability thresholds: how much proxy-true divergence a monitor
# must observe to flag the hack.
DETECT_THRESHOLDS = {
    "obvious": 0.5,    # easy to detect
    "subtle": 1.2,     # moderate difficulty
    "invisible": 999.0,  # monitors effectively cannot detect
}


def sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        return 1.0 / (1.0 + np.exp(-x))
    else:
        ez = np.exp(x)
        return ez / (1.0 + ez)


def run_simulation(
    agents: list[Agent],
    neighbors: list[list[int]],
    n_rounds: int,
    initial_hackers: list[int],
    t_discover: int,
    detectability: str,
    rng: np.random.Generator,
) -> dict:
    """Run one simulation and return time-series metrics.

    Parameters
    ----------
    agents : list of Agent
        The agent population.
    neighbors : list of list of int
        Adjacency list.
    n_rounds : int
        Number of rounds to simulate.
    initial_hackers : list of int
        Agent IDs that discover the hack at round t_discover.
    t_discover : int
        Round at which initial hackers switch to hacking.
    detectability : str
        One of "obvious", "subtle", "invisible".
    rng : numpy.random.Generator
        RNG for reproducibility.

    Returns
    -------
    result : dict with keys:
        "adoption_curve": list[float] — hack adoption fraction per round
        "proxy_reward_curve": list[float] — mean proxy reward per round
        "true_reward_curve": list[float] — mean true reward per round
        "divergence_curve": list[float] — mean |proxy - true| per round
        "containment_events": int — number of monitor quarantine actions
        "final_adoption": float — hack adoption at last round
        "time_to_50pct": int or None — first round with >=50% adoption
        "time_to_90pct": int or None — first round with >=90% adoption
    """
    n = len(agents)
    threshold = DETECT_THRESHOLDS[detectability]

    # Reset all agents
    for a in agents:
        a.strategy = "honest"
        a.proxy_reward = 0.0
        a.true_reward = 0.0
        a.rounds_hacking = 0
        a.divergence_history = []

    adoption_curve: list[float] = []
    proxy_curve: list[float] = []
    true_curve: list[float] = []
    divergence_curve: list[float] = []
    containment_events = 0
    time_to_50: int | None = None
    time_to_90: int | None = None

    for t in range(n_rounds):
        # --- Phase 0: inject hack at discovery time ---
        if t == t_discover:
            for idx in initial_hackers:
                agents[idx].strategy = "hack"

        # --- Phase 1: sample rewards ---
        for a in agents:
            a.proxy_reward, a.true_reward = sample_reward(a, rng)
            if a.is_hacking:
                a.rounds_hacking += 1

        # --- Phase 2: social learning (imitation) ---
        # Collect adoption decisions, then apply (synchronous update)
        new_strategies: list[str] = [a.strategy for a in agents]

        for a in agents:
            if a.agent_type == "monitor":
                continue  # monitors don't imitate
            if a.is_hacking:
                continue  # already hacking, no change

            nbrs = neighbors[a.agent_id]
            if not nbrs:
                continue

            # Find best neighbor by proxy reward
            best_nbr_idx = max(nbrs, key=lambda j: agents[j].proxy_reward)
            best_nbr = agents[best_nbr_idx]

            reward_gap = best_nbr.proxy_reward - a.proxy_reward
            if reward_gap > 0 and best_nbr.is_hacking:
                p_adopt = sigmoid(a.beta * reward_gap)
                if rng.random() < p_adopt:
                    new_strategies[a.agent_id] = "hack"

        # Apply synchronous update
        for i, a in enumerate(agents):
            a.strategy = new_strategies[i]

        # --- Phase 3: monitor detection & quarantine ---
        for a in agents:
            if a.agent_type != "monitor":
                continue
            nbrs = neighbors[a.agent_id]
            if not nbrs:
                continue
            # Monitor observes neighbors' proxy-true divergence
            divs = [abs(agents[j].proxy_reward - agents[j].true_reward) for j in nbrs]
            mean_div = float(np.mean(divs)) if divs else 0.0
            a.divergence_history.append(mean_div)

            if mean_div > threshold:
                # Quarantine: reset hacking neighbors to honest
                for j in nbrs:
                    if agents[j].is_hacking:
                        agents[j].strategy = "honest"
                        agents[j].rounds_hacking = 0
                        containment_events += 1

        # --- Record metrics ---
        # Adoption fraction measured against non-monitor agents only
        non_monitors = [a for a in agents if a.agent_type != "monitor"]
        n_non_monitors = len(non_monitors) if non_monitors else 1
        n_hacking = sum(1 for a in non_monitors if a.is_hacking)
        frac = n_hacking / n_non_monitors
        adoption_curve.append(frac)
        proxy_curve.append(float(np.mean([a.proxy_reward for a in agents])))
        true_curve.append(float(np.mean([a.true_reward for a in agents])))
        divergence_curve.append(
            float(np.mean([abs(a.proxy_reward - a.true_reward) for a in agents]))
        )

        if time_to_50 is None and frac >= 0.5:
            time_to_50 = t
        if time_to_90 is None and frac >= 0.9:
            time_to_90 = t

    return {
        "adoption_curve": adoption_curve,
        "proxy_reward_curve": proxy_curve,
        "true_reward_curve": true_curve,
        "divergence_curve": divergence_curve,
        "containment_events": containment_events,
        "final_adoption": adoption_curve[-1] if adoption_curve else 0.0,
        "time_to_50pct": time_to_50,
        "time_to_90pct": time_to_90,
    }
