"""Simulation engine for multi-agent model collapse experiments.

Runs generational data loops: each generation of agents learns from the
previous generation's synthetic output (mixed with ground truth according
to gt_fraction), then produces data for the next generation.

The gt_fraction is applied at the DATA PIPELINE level: each generation's
training set is composed of (1 - gt_fraction) synthetic samples from the
previous generation plus gt_fraction fresh ground-truth samples.  This
affects ALL agent types equally.  The Anchored agent additionally mixes
in ground truth in its own learn() method, providing a double anchor.

Uses multiprocessing to parallelise independent simulation configs.
"""

from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .agents import AGENT_CLASSES, SAMPLES_PER_GENERATION, REFERENCE_SAMPLES
from .distributions import (
    kl_divergence_numerical,
    sample_ground_truth,
    wasserstein,
)


@dataclass
class SimConfig:
    """One simulation configuration."""

    agent_type: str
    gt_fraction: float
    dist_name: str
    seed: int
    n_generations: int = 10


@dataclass
class GenerationRecord:
    """Metrics for a single generation."""

    generation: int
    kl_divergence: float
    wasserstein_distance: float


@dataclass
class SimResult:
    """Full result of one simulation run."""

    config: SimConfig
    generations: list[GenerationRecord] = field(default_factory=list)
    collapse_generation: int | None = None  # first gen where KL > threshold


KL_COLLAPSE_THRESHOLD = 1.0  # nats


def _run_single(config: SimConfig) -> dict[str, Any]:
    """Execute one simulation and return a serialisable dict.

    This is the top-level function called by the worker pool.
    It must be module-level and picklable.
    """
    rng = np.random.default_rng(config.seed)

    agent_cls = AGENT_CLASSES[config.agent_type]
    agent = agent_cls(
        dist_name=config.dist_name,
        gt_fraction=config.gt_fraction,
        rng=rng,
    )

    # Generation 0: train on ground-truth data
    training_data = sample_ground_truth(config.dist_name, SAMPLES_PER_GENERATION, rng)
    ref_samples = sample_ground_truth(config.dist_name, REFERENCE_SAMPLES, rng)

    generations: list[dict[str, Any]] = []
    collapse_gen: int | None = None

    for gen in range(config.n_generations):
        agent.learn(training_data)

        kl = kl_divergence_numerical(config.dist_name, agent.kde)
        synth = agent.produce(SAMPLES_PER_GENERATION)
        wd = wasserstein(ref_samples, synth)

        generations.append({
            "generation": gen,
            "kl_divergence": float(kl),
            "wasserstein_distance": float(wd),
        })

        if collapse_gen is None and kl > KL_COLLAPSE_THRESHOLD:
            collapse_gen = gen

        # Build next generation's training data:
        # Mix synthetic output with fresh ground truth according to gt_fraction
        n_total = SAMPLES_PER_GENERATION
        n_gt = int(config.gt_fraction * n_total)
        n_synth = n_total - n_gt

        if n_gt > 0:
            gt_fresh = sample_ground_truth(config.dist_name, n_gt, rng)
            # Take a random subset of synthetic data
            if n_synth > 0 and n_synth < len(synth):
                idx = rng.choice(len(synth), size=n_synth, replace=False)
                synth_subset = synth[idx]
            else:
                synth_subset = synth
            training_data = np.concatenate([synth_subset, gt_fresh])
        else:
            training_data = synth

    return {
        "config": {
            "agent_type": config.agent_type,
            "gt_fraction": config.gt_fraction,
            "dist_name": config.dist_name,
            "seed": config.seed,
            "n_generations": config.n_generations,
        },
        "generations": generations,
        "collapse_generation": collapse_gen,
    }


def build_configs(
    agent_types: list[str] | None = None,
    gt_fractions: list[float] | None = None,
    dist_names: list[str] | None = None,
    seeds: list[int] | None = None,
    n_generations: int = 10,
) -> list[SimConfig]:
    """Build the full grid of simulation configs."""
    if agent_types is None:
        agent_types = ["naive", "selective", "anchored"]
    if gt_fractions is None:
        gt_fractions = [0.0, 0.01, 0.05, 0.10, 0.50]
    if dist_names is None:
        dist_names = ["bimodal", "skewed", "uniform_like"]
    if seeds is None:
        seeds = [42, 123, 789]

    configs = []
    for at in agent_types:
        for gf in gt_fractions:
            for dn in dist_names:
                for s in seeds:
                    configs.append(SimConfig(
                        agent_type=at,
                        gt_fraction=gf,
                        dist_name=dn,
                        seed=s,
                        n_generations=n_generations,
                    ))
    return configs


def run_experiment(
    configs: list[SimConfig] | None = None,
    n_workers: int | None = None,
) -> list[dict[str, Any]]:
    """Run all simulations in parallel and return results.

    Parameters
    ----------
    configs : list of SimConfig, optional
        If None, builds the default 135-config grid.
    n_workers : int, optional
        Number of parallel workers.  Defaults to min(cpu_count, 8).
    """
    if configs is None:
        configs = build_configs()

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    # Use spawn context for safety on macOS
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_workers) as pool:
        results = pool.map(_run_single, configs)

    return results
