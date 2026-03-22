"""Tests for the experiment runner."""

import numpy as np
import pytest

from src.experiment import SimConfig, SimResult, run_simulation, build_experiment_matrix


class TestSimConfig:
    def test_label_format(self):
        cfg = SimConfig(
            learner_code="NL",
            adversary_code="SA",
            drift_regime="stable",
            noise_level=0.0,
            seed=42,
        )
        assert cfg.label == "NL-vs-SA_stable_noise0.0_s42"

    def test_default_n_rounds(self):
        cfg = SimConfig(
            learner_code="NL",
            adversary_code="RA",
            drift_regime="stable",
            noise_level=0.0,
            seed=0,
        )
        assert cfg.n_rounds == 50_000


class TestRunSimulation:
    def test_short_sim_returns_result(self):
        cfg = SimConfig(
            learner_code="NL",
            adversary_code="RA",
            drift_regime="stable",
            noise_level=0.0,
            seed=0,
            n_rounds=200,
            belief_sample_interval=10,
        )
        result = run_simulation(cfg)
        assert isinstance(result, SimResult)
        assert result.config is cfg
        assert "distortion" in result.audit
        assert "decision_quality" in result.audit
        assert len(result.belief_error_timeseries) == 20  # 200 / 10

    def test_reproducibility(self):
        cfg = SimConfig(
            learner_code="SL",
            adversary_code="SA",
            drift_regime="volatile",
            noise_level=0.1,
            seed=99,
            n_rounds=500,
        )
        r1 = run_simulation(cfg)
        r2 = run_simulation(cfg)
        assert r1.audit["distortion"]["mean_belief_error"] == pytest.approx(
            r2.audit["distortion"]["mean_belief_error"]
        )

    def test_al_has_trust_timeseries(self):
        cfg = SimConfig(
            learner_code="AL",
            adversary_code="PA",
            drift_regime="stable",
            noise_level=0.0,
            seed=0,
            n_rounds=100,
            belief_sample_interval=10,
        )
        result = run_simulation(cfg)
        assert result.trust_timeseries is not None
        assert len(result.trust_timeseries) == 10

    def test_nl_has_no_trust_timeseries(self):
        cfg = SimConfig(
            learner_code="NL",
            adversary_code="RA",
            drift_regime="stable",
            noise_level=0.0,
            seed=0,
            n_rounds=100,
        )
        result = run_simulation(cfg)
        assert result.trust_timeseries is None

    def test_sa_distorts_more_than_ra(self):
        """NL-vs-SA should have higher belief error than NL-vs-RA."""
        cfg_ra = SimConfig(
            learner_code="NL",
            adversary_code="RA",
            drift_regime="stable",
            noise_level=0.0,
            seed=42,
            n_rounds=1000,
        )
        cfg_sa = SimConfig(
            learner_code="NL",
            adversary_code="SA",
            drift_regime="stable",
            noise_level=0.0,
            seed=42,
            n_rounds=1000,
        )
        r_ra = run_simulation(cfg_ra)
        r_sa = run_simulation(cfg_sa)
        assert (
            r_sa.audit["distortion"]["mean_belief_error"]
            > r_ra.audit["distortion"]["mean_belief_error"]
        )


class TestBuildExperimentMatrix:
    def test_matrix_size(self):
        configs = build_experiment_matrix(n_rounds=100, seeds=[0, 1, 2])
        # 3 learners x 3 adversaries x 3 regimes x 2 noise x 3 seeds = 162
        assert len(configs) == 162

    def test_custom_seeds(self):
        configs = build_experiment_matrix(n_rounds=100, seeds=[10, 20])
        assert len(configs) == 108  # 3*3*3*2*2 = 108

    def test_all_configs_valid(self):
        configs = build_experiment_matrix(n_rounds=100, seeds=[0])
        for cfg in configs:
            assert cfg.learner_code in ("NL", "SL", "AL")
            assert cfg.adversary_code in ("RA", "SA", "PA")
            assert cfg.drift_regime in ("stable", "slow_drift", "volatile")
            assert cfg.noise_level in (0.0, 0.1)
