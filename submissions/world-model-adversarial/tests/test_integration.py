"""Integration tests: end-to-end simulation and metric ordering."""

import numpy as np
import pytest

from src.experiment import SimConfig, run_simulation
from src.analysis import aggregate_results, compute_manipulation_speed


class TestEndToEnd:
    """Run short simulations and verify structural properties."""

    def test_sa_vs_ra_for_naive_learner(self):
        """SA should produce higher belief error than RA for NL."""
        results = []
        for ac in ["RA", "SA"]:
            cfg = SimConfig(
                learner_code="NL", adversary_code=ac,
                drift_regime="stable", noise_level=0.0, seed=0,
                n_rounds=2000,
            )
            results.append(run_simulation(cfg))
        ra_err = results[0].audit["distortion"]["mean_belief_error"]
        sa_err = results[1].audit["distortion"]["mean_belief_error"]
        assert sa_err > ra_err

    def test_sa_vs_ra_volatile(self):
        """Same ordering holds in volatile environment."""
        results = []
        for ac in ["RA", "SA"]:
            cfg = SimConfig(
                learner_code="NL", adversary_code=ac,
                drift_regime="volatile", noise_level=0.0, seed=0,
                n_rounds=2000,
            )
            results.append(run_simulation(cfg))
        ra_err = results[0].audit["distortion"]["mean_belief_error"]
        sa_err = results[1].audit["distortion"]["mean_belief_error"]
        assert sa_err > ra_err

    def test_pa_has_lower_early_error_than_late(self):
        """PA's credibility phase should produce lower early error."""
        cfg = SimConfig(
            learner_code="NL", adversary_code="PA",
            drift_regime="stable", noise_level=0.0, seed=0,
            n_rounds=2000,
        )
        result = run_simulation(cfg)
        early = result.audit["credibility"]["early_belief_error"]
        late = result.audit["credibility"]["late_belief_error"]
        assert early < late

    def test_noise_increases_baseline_error(self):
        """Noisy signals should increase error even with RA."""
        results = {}
        for noise in [0.0, 0.1]:
            cfg = SimConfig(
                learner_code="NL", adversary_code="RA",
                drift_regime="stable", noise_level=noise, seed=0,
                n_rounds=2000,
            )
            results[noise] = run_simulation(cfg)
        clean_acc = results[0.0].audit["decision_quality"]["accuracy"]
        noisy_acc = results[0.1].audit["decision_quality"]["accuracy"]
        # Clean should be at least as accurate as noisy.
        assert clean_acc >= noisy_acc - 0.05  # Small tolerance for stochasticity.

    def test_aggregate_groups_count(self):
        """Aggregation across seeds produces correct group count."""
        results = []
        for seed in [0, 1]:
            for ac in ["RA", "SA"]:
                cfg = SimConfig(
                    learner_code="NL", adversary_code=ac,
                    drift_regime="stable", noise_level=0.0, seed=seed,
                    n_rounds=200,
                )
                results.append(run_simulation(cfg))
        agg = aggregate_results(results)
        assert len(agg) == 2  # NL-vs-RA_stable_noise0.0, NL-vs-SA_stable_noise0.0

    def test_manipulation_speed_sa_faster_than_ra(self):
        """SA should reach distortion threshold faster than RA."""
        results = []
        for ac in ["RA", "SA"]:
            for seed in [0, 1]:
                cfg = SimConfig(
                    learner_code="NL", adversary_code=ac,
                    drift_regime="stable", noise_level=0.0, seed=seed,
                    n_rounds=2000,
                )
                results.append(run_simulation(cfg))
        speeds = compute_manipulation_speed(results, threshold=0.8)
        key_ra = "NL-vs-RA_stable_noise0.0"
        key_sa = "NL-vs-SA_stable_noise0.0"
        assert speeds[key_sa]["mean_rounds"] <= speeds[key_ra]["mean_rounds"]

    def test_belief_error_bounded(self):
        """All belief errors should be in [0, 1]."""
        cfg = SimConfig(
            learner_code="AL", adversary_code="PA",
            drift_regime="volatile", noise_level=0.1, seed=42,
            n_rounds=1000,
        )
        result = run_simulation(cfg)
        for err in result.belief_error_timeseries:
            assert 0.0 <= err <= 1.0 + 1e-10

    def test_al_trust_changes_over_time(self):
        """AL's trust timeseries should not be constant."""
        cfg = SimConfig(
            learner_code="AL", adversary_code="PA",
            drift_regime="stable", noise_level=0.0, seed=0,
            n_rounds=2000, belief_sample_interval=10,
        )
        result = run_simulation(cfg)
        assert result.trust_timeseries is not None
        ts = np.array(result.trust_timeseries)
        # Trust should change over time (not all identical).
        assert ts.std() > 0.001
