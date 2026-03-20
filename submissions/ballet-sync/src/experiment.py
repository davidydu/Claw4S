"""Experiment runner for Kuramoto synchronization K-sweep."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.kuramoto import KuramotoModel

# K sweep: 20 values from 0.0 to 2.85
K_RANGE = np.linspace(0.0, 2.85, 20)


@dataclass
class ExperimentConfig:
    """Configuration for a single Kuramoto simulation run."""

    K: float
    topology: str
    n: int
    sigma: float
    seed: int
    total_steps: int = 10_000
    dt: float = 0.01
    omega0: float = 1.0
    stage_size: float = 10.0
    topology_kwargs: Optional[dict] = field(default=None)


@dataclass
class SimulationResult:
    """Results from a completed simulation run."""

    config: ExperimentConfig
    phase_history: np.ndarray          # shape (T, N)
    final_r: float
    convergence_step: Optional[int]    # first step where r > 0.5 and stays
    evaluator_results: Optional[list] = None


def run_simulation(config: ExperimentConfig) -> SimulationResult:
    """Run a Kuramoto simulation and return results.

    Creates a KuramotoModel from config parameters, runs total_steps RK4
    iterations recording phase history at every step, then computes final_r
    as the mean order parameter over the last 20% of steps and detects
    convergence as the first step where r exceeds 0.5 and remains above it.
    """
    model = KuramotoModel(
        n=config.n,
        K=config.K,
        sigma=config.sigma,
        omega0=config.omega0,
        topology=config.topology,
        dt=config.dt,
        stage_size=config.stage_size,
        seed=config.seed,
        topology_kwargs=config.topology_kwargs,
    )

    T = config.total_steps
    N = config.n
    phase_history = np.empty((T, N), dtype=float)

    for t in range(T):
        phase_history[t] = model.phases
        model.step()

    # Compute order parameter for every step
    order_params = np.array([
        model.compute_order_parameter(phase_history[t]) for t in range(T)
    ])

    # final_r: mean over last 20% of steps
    tail_start = int(0.8 * T)
    final_r = float(np.mean(order_params[tail_start:]))

    # Convergence: first step where r > 0.5 and all subsequent steps also > 0.5
    convergence_step: Optional[int] = None
    for t in range(T):
        if order_params[t] > 0.5 and np.all(order_params[t:] > 0.5):
            convergence_step = t
            break

    return SimulationResult(
        config=config,
        phase_history=phase_history,
        final_r=final_r,
        convergence_step=convergence_step,
    )
