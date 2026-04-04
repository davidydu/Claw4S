"""Tests for experiment configuration and runner."""

from src.experiment import (
    ExperimentConfig,
    ExperimentResult,
    build_experiment_matrix,
    run_simulation,
    COMPOSITIONS,
    MARKET_SIZES,
    INFO_REGIMES,
    SEEDS,
)


class TestExperimentConfig:
    def test_name_format(self):
        c = ExperimentConfig("all_honest", "small", "opaque", 42)
        assert c.name == "all_honest__small__opaque__seed42"

    def test_matrix_size(self):
        configs = build_experiment_matrix()
        expected = len(COMPOSITIONS) * len(MARKET_SIZES) * len(INFO_REGIMES) * len(SEEDS)
        assert len(configs) == expected
        assert expected == 162


class TestRunSimulation:
    def test_small_honest_opaque(self):
        """Smoke test: run a small, fast simulation."""
        c = ExperimentConfig("all_honest", "small", "opaque", 42, n_rounds=50)
        result = run_simulation(c)
        assert isinstance(result, ExperimentResult)
        assert result.metrics["n_transactions"] > 0
        assert 0 <= result.metrics["market_efficiency"] <= 1.0

    def test_mixed_sellers_transparent(self):
        c = ExperimentConfig("mixed_sellers", "medium", "transparent", 42, n_rounds=50)
        result = run_simulation(c)
        assert len(result.buyer_welfare) == 3  # 3 buyers
        assert len(result.seller_profit) == 3  # 3 sellers

    def test_all_predatory_high_lemons(self):
        c = ExperimentConfig("all_predatory", "medium", "opaque", 42, n_rounds=100)
        result = run_simulation(c)
        assert result.metrics["lemons_index"] == 1.0  # all sellers are low-quality

    def test_audit_scores_present(self):
        c = ExperimentConfig("all_honest", "small", "opaque", 42, n_rounds=50)
        result = run_simulation(c)
        assert "fair_pricing" in result.audit_scores
        assert "exploitation" in result.audit_scores
        assert "market_efficiency" in result.audit_scores
        assert "information_asymmetry" in result.audit_scores

    def test_reproducible(self):
        """Same config + seed → same results."""
        c = ExperimentConfig("mixed_sellers", "small", "opaque", 42, n_rounds=50)
        r1 = run_simulation(c)
        r2 = run_simulation(c)
        assert r1.metrics == r2.metrics
