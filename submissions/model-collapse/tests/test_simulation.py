"""Tests for the simulation engine."""

import numpy as np
import pytest

from src.simulation import SimConfig, _run_single, build_configs, run_experiment


class TestSimConfig:
    """SimConfig construction."""

    def test_defaults(self) -> None:
        c = SimConfig("naive", 0.0, "bimodal", 42)
        assert c.n_generations == 10

    def test_build_configs_default_grid(self) -> None:
        configs = build_configs()
        # 3 agents x 5 fractions x 3 distributions x 3 seeds = 135
        assert len(configs) == 135

    def test_build_configs_custom(self) -> None:
        configs = build_configs(
            agent_types=["naive"],
            gt_fractions=[0.0],
            dist_names=["bimodal"],
            seeds=[42],
        )
        assert len(configs) == 1


class TestRunSingle:
    """Single simulation execution."""

    def test_basic_run(self) -> None:
        config = SimConfig("naive", 0.0, "bimodal", 42, n_generations=3)
        result = _run_single(config)
        assert len(result["generations"]) == 3
        assert result["config"]["agent_type"] == "naive"
        for g in result["generations"]:
            assert "kl_divergence" in g
            assert "wasserstein_distance" in g
            assert g["kl_divergence"] >= 0

    def test_reproducibility(self) -> None:
        config = SimConfig("naive", 0.0, "bimodal", 42, n_generations=5)
        r1 = _run_single(config)
        r2 = _run_single(config)
        for g1, g2 in zip(r1["generations"], r2["generations"]):
            assert g1["kl_divergence"] == g2["kl_divergence"]
            assert g1["wasserstein_distance"] == g2["wasserstein_distance"]

    def test_collapse_detected(self) -> None:
        """Naive agent should eventually report collapse."""
        config = SimConfig("naive", 0.0, "bimodal", 42, n_generations=100)
        result = _run_single(config)
        assert result["collapse_generation"] is not None
        assert result["collapse_generation"] < 100


class TestRunExperiment:
    """Parallel experiment execution."""

    def test_small_parallel(self) -> None:
        configs = build_configs(
            agent_types=["naive", "anchored"],
            gt_fractions=[0.0, 0.05],
            dist_names=["bimodal"],
            seeds=[42],
            n_generations=3,
        )
        results = run_experiment(configs, n_workers=2)
        assert len(results) == 4
        for r in results:
            assert len(r["generations"]) == 3
