"""Tests for the simulation engine."""

import pytest
from src.simulation import SimConfig, run_single_sim


class TestSimConfig:
    def test_label(self):
        cfg = SimConfig(
            scheme_name="fixed_pay",
            worker_types=["honest", "shirker", "strategic"],
            noise_std=0.5,
            seed=42,
        )
        assert "fixed_pay" in cfg.label
        assert "noise0.5" in cfg.label


class TestRunSingleSim:
    def test_honest_workers_no_shirking(self):
        cfg = SimConfig(
            scheme_name="piece_rate",
            worker_types=["honest", "honest", "honest"],
            noise_std=0.5,
            num_rounds=500,
            seed=42,
        )
        result = run_single_sim(cfg)
        assert result.shirking_rate == 0.0
        assert result.avg_quality > 4.0  # effort=5 + small noise

    def test_shirker_workers_all_shirk(self):
        cfg = SimConfig(
            scheme_name="fixed_pay",
            worker_types=["shirker", "shirker", "shirker"],
            noise_std=0.5,
            num_rounds=500,
            seed=42,
        )
        result = run_single_sim(cfg)
        assert result.shirking_rate == 1.0
        assert result.avg_quality < 2.0  # effort=1 + small noise

    def test_metrics_in_range(self):
        cfg = SimConfig(
            scheme_name="tournament",
            worker_types=["honest", "strategic", "adaptive"],
            noise_std=1.0,
            num_rounds=1000,
            seed=42,
        )
        result = run_single_sim(cfg)
        assert 0.0 < result.avg_quality < 10.0
        assert 0.0 <= result.shirking_rate <= 1.0
        assert result.incentive_efficiency > 0
        assert result.quality_variance >= 0

    def test_deterministic(self):
        cfg = SimConfig(
            scheme_name="piece_rate",
            worker_types=["strategic", "adaptive", "honest"],
            noise_std=1.0,
            num_rounds=500,
            seed=99,
        )
        r1 = run_single_sim(cfg)
        r2 = run_single_sim(cfg)
        assert r1.avg_quality == pytest.approx(r2.avg_quality)
        assert r1.shirking_rate == pytest.approx(r2.shirking_rate)

    def test_per_worker_output(self):
        cfg = SimConfig(
            scheme_name="reputation",
            worker_types=["honest", "shirker"],
            noise_std=0.5,
            num_rounds=500,
            seed=42,
        )
        result = run_single_sim(cfg)
        assert len(result.per_worker) == 2
        for wname, wdata in result.per_worker.items():
            assert "avg_effort" in wdata
            assert "avg_wage" in wdata
            assert "type" in wdata

    def test_to_dict(self):
        cfg = SimConfig(
            scheme_name="fixed_pay",
            worker_types=["honest"],
            noise_std=0.5,
            num_rounds=100,
            seed=42,
        )
        result = run_single_sim(cfg)
        d = result.to_dict()
        assert d["scheme"] == "fixed_pay"
        assert isinstance(d["avg_quality"], float)
        assert isinstance(d["per_worker"], dict)


class TestNoiseEffect:
    def test_high_noise_increases_variance(self):
        cfg_low = SimConfig(
            scheme_name="piece_rate",
            worker_types=["honest", "honest", "honest"],
            noise_std=0.1,
            num_rounds=2000,
            seed=42,
        )
        cfg_high = SimConfig(
            scheme_name="piece_rate",
            worker_types=["honest", "honest", "honest"],
            noise_std=3.0,
            num_rounds=2000,
            seed=42,
        )
        r_low = run_single_sim(cfg_low)
        r_high = run_single_sim(cfg_high)
        assert r_high.quality_variance > r_low.quality_variance
