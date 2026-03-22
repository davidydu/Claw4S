"""Tests for sweep orchestration."""

from src.sweep import run_single


class TestRunSingle:
    """Tests for individual training runs."""

    def test_returns_expected_keys(self):
        """Result should contain config, metrics, phase, grokking_gap."""
        result = run_single(
            p=5, embed_dim=4, hidden_dim=8,
            weight_decay=0.1, train_fraction=0.7,
            max_epochs=100, seed=42,
        )
        assert "config" in result
        assert "metrics" in result
        assert "phase" in result
        assert "grokking_gap" in result
        assert "elapsed_seconds" in result

    def test_config_stored(self):
        """Config should be stored in the result."""
        result = run_single(
            p=5, embed_dim=4, hidden_dim=8,
            weight_decay=0.1, train_fraction=0.7,
            max_epochs=100, seed=42,
        )
        assert result["config"]["p"] == 5
        assert result["config"]["hidden_dim"] == 8
        assert result["config"]["weight_decay"] == 0.1
        assert result["config"]["train_fraction"] == 0.7
        assert result["config"]["param_count"] > 0

    def test_metrics_populated(self):
        """Metrics should have accuracy and loss lists."""
        result = run_single(
            p=5, embed_dim=4, hidden_dim=8,
            weight_decay=0.1, train_fraction=0.7,
            max_epochs=200, seed=42,
        )
        m = result["metrics"]
        assert len(m["train_accs"]) > 0
        assert len(m["test_accs"]) > 0
        assert 0 <= m["final_train_acc"] <= 1.0
        assert 0 <= m["final_test_acc"] <= 1.0

    def test_phase_is_valid_string(self):
        """Phase should be one of the four valid values."""
        result = run_single(
            p=5, embed_dim=4, hidden_dim=8,
            weight_decay=0.1, train_fraction=0.7,
            max_epochs=100, seed=42,
        )
        assert result["phase"] in [
            "confusion", "memorization", "grokking", "comprehension"
        ]

    def test_deterministic(self):
        """Same seed should produce same results."""
        r1 = run_single(
            p=5, embed_dim=4, hidden_dim=8,
            weight_decay=0.1, train_fraction=0.7,
            max_epochs=100, seed=42,
        )
        r2 = run_single(
            p=5, embed_dim=4, hidden_dim=8,
            weight_decay=0.1, train_fraction=0.7,
            max_epochs=100, seed=42,
        )
        assert r1["metrics"]["final_train_acc"] == r2["metrics"]["final_train_acc"]
        assert r1["metrics"]["final_test_acc"] == r2["metrics"]["final_test_acc"]
        assert r1["phase"] == r2["phase"]
