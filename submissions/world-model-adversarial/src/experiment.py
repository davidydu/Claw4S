"""Experiment runner for the adversarial signaling game.

Runs a single simulation of a learner--adversary matchup in a given
environment and returns a :class:`SimTrace` plus audit results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.agents import Learner, Adversary, make_learner, make_adversary
from src.auditors import SimTrace, run_all_auditors
from src.environment import HiddenEnvironment, DriftRegime


@dataclass(frozen=True)
class SimConfig:
    """Configuration for a single simulation run.

    Attributes
    ----------
    learner_code : str
        Learner short code (NL, SL, AL).
    adversary_code : str
        Adversary short code (RA, SA, PA).
    drift_regime : DriftRegime
        Environment drift regime.
    noise_level : float
        Signal noise level (0.0 = clean, 0.1 = noisy).
    seed : int
        Random seed for reproducibility.
    n_rounds : int
        Number of game rounds.
    n_states : int
        Number of discrete environment states.
    belief_sample_interval : int
        Store beliefs every N rounds to reduce memory.
    """

    learner_code: str
    adversary_code: str
    drift_regime: DriftRegime
    noise_level: float
    seed: int
    n_rounds: int = 50_000
    n_states: int = 5
    belief_sample_interval: int = 10

    @property
    def label(self) -> str:
        return (
            f"{self.learner_code}-vs-{self.adversary_code}"
            f"_{self.drift_regime}_noise{self.noise_level}_s{self.seed}"
        )


@dataclass
class SimResult:
    """Result of a single simulation.

    Attributes
    ----------
    config : SimConfig
        The configuration that produced this result.
    audit : dict[str, dict[str, float]]
        Audit metrics from all auditors.
    belief_error_timeseries : list[float]
        Sampled belief error time series (at ``belief_sample_interval``).
    trust_timeseries : list[float] | None
        Trust time series for AdaptiveLearner, else None.
    """

    config: SimConfig
    audit: dict[str, dict[str, float]]
    belief_error_timeseries: list[float]
    trust_timeseries: list[float] | None


def run_simulation(config: SimConfig) -> SimResult:
    """Run a single simulation and return results.

    This function is designed to be called from a multiprocessing Pool.
    """
    # Deterministic RNG hierarchy: split from config seed.
    master_rng = np.random.default_rng(config.seed)
    env_seed, learner_seed, adversary_seed = master_rng.integers(
        0, 2**31, size=3
    )

    env = HiddenEnvironment(
        n_states=config.n_states,
        drift_regime=config.drift_regime,
        seed=int(env_seed),
    )
    learner = make_learner(
        config.learner_code, config.n_states,
        rng=np.random.default_rng(int(learner_seed)),
    )
    adversary = make_adversary(
        config.adversary_code, config.n_states,
        rng=np.random.default_rng(int(adversary_seed)),
    )

    # Pre-allocate trace arrays.
    true_states = np.empty(config.n_rounds, dtype=np.int64)
    signals = np.empty(config.n_rounds, dtype=np.int64)
    actions = np.empty(config.n_rounds, dtype=np.int64)
    beliefs = np.empty((config.n_rounds, config.n_states), dtype=np.float64)

    # Sampled time series for lightweight storage.
    belief_error_ts: list[float] = []
    trust_ts: list[float] | None = (
        [] if config.learner_code == "AL" else None
    )

    for t in range(config.n_rounds):
        ts = env.true_state
        sig = adversary.choose_signal(ts, learner.beliefs)
        sig = env.generate_noisy_signal(sig, config.noise_level)
        learner.update(sig)
        act = learner.choose_action()

        true_states[t] = ts
        signals[t] = sig
        actions[t] = act
        beliefs[t] = learner.beliefs.copy()

        # Sample time series.
        if t % config.belief_sample_interval == 0:
            belief_error_ts.append(float(1.0 - learner.beliefs[ts]))
            if trust_ts is not None:
                trust_ts.append(float(learner.trust))  # type: ignore[attr-defined]

        env.step()

    trace = SimTrace(
        n_rounds=config.n_rounds,
        true_states=true_states,
        signals=signals,
        actions=actions,
        beliefs=beliefs,
        n_states=config.n_states,
    )

    audit = run_all_auditors(trace)

    return SimResult(
        config=config,
        audit=audit,
        belief_error_timeseries=belief_error_ts,
        trust_timeseries=trust_ts,
    )


def build_experiment_matrix(
    n_rounds: int = 50_000,
    n_states: int = 5,
    seeds: list[int] | None = None,
    belief_sample_interval: int = 10,
) -> list[SimConfig]:
    """Build the full experiment matrix.

    9 matchups x 3 environments x 2 noise levels x 3 seeds = 162 configs.
    """
    if seeds is None:
        seeds = [0, 1, 2]

    learner_codes = ["NL", "SL", "AL"]
    adversary_codes = ["RA", "SA", "PA"]
    drift_regimes: list[DriftRegime] = ["stable", "slow_drift", "volatile"]
    noise_levels = [0.0, 0.1]

    configs: list[SimConfig] = []
    for lc in learner_codes:
        for ac in adversary_codes:
            for dr in drift_regimes:
                for nl in noise_levels:
                    for seed in seeds:
                        configs.append(SimConfig(
                            learner_code=lc,
                            adversary_code=ac,
                            drift_regime=dr,
                            noise_level=nl,
                            seed=seed,
                            n_rounds=n_rounds,
                            n_states=n_states,
                            belief_sample_interval=belief_sample_interval,
                        ))
    return configs
