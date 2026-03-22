"""Core simulation engine for the delegation game.

Runs one simulation: a principal with an incentive scheme delegates tasks
to N workers over T rounds. Each round:
  1. Workers choose effort (1-5)
  2. Output quality = effort + Gaussian noise
  3. Principal observes quality, pays wages per incentive scheme
  4. Metrics are recorded
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Any

from src.workers import Worker, create_worker
from src.incentives import IncentiveScheme, create_scheme


@dataclass
class SimConfig:
    """Configuration for a single simulation run."""
    scheme_name: str
    worker_types: list[str]  # one per worker
    noise_std: float  # std dev of quality noise
    num_rounds: int = 10_000
    seed: int = 42

    @property
    def label(self) -> str:
        worker_str = "-".join(sorted(self.worker_types))
        return f"{self.scheme_name}__{worker_str}__noise{self.noise_std}"


@dataclass
class SimResult:
    """Aggregated results from one simulation run."""
    config: SimConfig
    avg_quality: float
    principal_net_payoff: float
    worker_surplus: float
    shirking_rate: float
    quality_variance: float
    incentive_efficiency: float
    per_worker: dict[str, dict[str, float]] = field(default_factory=dict)
    effort_trajectory: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.config.scheme_name,
            "worker_types": self.config.worker_types,
            "noise_std": self.config.noise_std,
            "seed": self.config.seed,
            "avg_quality": round(self.avg_quality, 4),
            "principal_net_payoff": round(self.principal_net_payoff, 4),
            "worker_surplus": round(self.worker_surplus, 4),
            "shirking_rate": round(self.shirking_rate, 4),
            "quality_variance": round(self.quality_variance, 4),
            "incentive_efficiency": round(self.incentive_efficiency, 4),
            "per_worker": {
                k: {mk: round(mv, 4) if isinstance(mv, float) else mv
                    for mk, mv in v.items()}
                for k, v in self.per_worker.items()
            },
        }


def run_single_sim(config: SimConfig) -> SimResult:
    """Run one simulation and return aggregated metrics."""
    rng = np.random.default_rng(config.seed)

    # Create workers
    workers: list[Worker] = []
    for i, wtype in enumerate(config.worker_types):
        w = create_worker(wtype, f"worker_{i}_{wtype}", rng=rng)
        workers.append(w)

    # Create incentive scheme
    scheme = create_scheme(config.scheme_name)

    # Tracking arrays
    all_qualities: list[float] = []
    all_wages: list[float] = []
    all_efforts: list[int] = []
    per_worker_data: dict[str, dict] = {
        w.name: {"efforts": [], "wages": [], "qualities": []}
        for w in workers
    }

    history: list[dict] = []
    # Only keep last 50 history entries to avoid memory blowup
    history_window = 50

    for t in range(config.num_rounds):
        # Workers choose effort
        efforts = []
        for w in workers:
            e = w.choose_effort(t, history[-history_window:])
            e = int(np.clip(e, 1, 5))
            efforts.append(e)

        # Generate output quality = effort + noise
        noise = rng.normal(0, config.noise_std, size=len(workers))
        qualities = [float(e) + float(n) for e, n in zip(efforts, noise)]

        # Principal pays based on observed quality
        worker_names = [w.name for w in workers]
        wages = scheme.compute_wages(qualities, worker_names, t)

        # Record round
        for i, w in enumerate(workers):
            entry = {
                "worker": w.name,
                "effort": efforts[i],
                "quality": qualities[i],
                "wage": wages[i],
                "round": t,
            }
            history.append(entry)
            all_qualities.append(qualities[i])
            all_wages.append(wages[i])
            all_efforts.append(efforts[i])
            per_worker_data[w.name]["efforts"].append(efforts[i])
            per_worker_data[w.name]["wages"].append(wages[i])
            per_worker_data[w.name]["qualities"].append(qualities[i])

    # Compute aggregate metrics
    q_arr = np.array(all_qualities)
    w_arr = np.array(all_wages)
    e_arr = np.array(all_efforts)

    avg_quality = float(np.mean(q_arr))
    total_quality = float(np.sum(q_arr))
    total_wages = float(np.sum(w_arr))
    principal_net_payoff = total_quality - total_wages
    effort_costs = float(np.sum(e_arr * 1.0))  # cost = effort * 1.0
    worker_surplus = total_wages - effort_costs
    shirking_rate = float(np.mean(e_arr < 3))
    quality_variance = float(np.var(q_arr))
    incentive_efficiency = (
        avg_quality / (total_wages / len(all_qualities))
        if total_wages > 0 else 0.0
    )

    # Per-worker metrics
    per_worker_results = {}
    for w in workers:
        wd = per_worker_data[w.name]
        ea = np.array(wd["efforts"])
        wa = np.array(wd["wages"])
        qa = np.array(wd["qualities"])
        per_worker_results[w.name] = {
            "type": w.worker_type,
            "avg_effort": float(np.mean(ea)),
            "avg_wage": float(np.mean(wa)),
            "avg_quality": float(np.mean(qa)),
            "shirking_rate": float(np.mean(ea < 3)),
        }

    return SimResult(
        config=config,
        avg_quality=avg_quality,
        principal_net_payoff=principal_net_payoff,
        worker_surplus=worker_surplus,
        shirking_rate=shirking_rate,
        quality_variance=quality_variance,
        incentive_efficiency=incentive_efficiency,
        per_worker=per_worker_results,
    )
