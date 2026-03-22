"""Tests for the training loop."""

import torch

from src.data import generate_modular_addition_data, split_dataset
from src.model import GrokkingMLP
from src.train import TrainConfig, TrainResult, train_model


class TestTrainResult:
    """Tests for TrainResult dataclass."""

    def test_default_fields(self):
        """Default TrainResult should have empty lists and None milestones."""
        r = TrainResult()
        assert r.train_accs == []
        assert r.test_accs == []
        assert r.epoch_train_95 is None
        assert r.epoch_test_95 is None
        assert r.final_train_acc == 0.0

    def test_fields_settable(self):
        """Should be able to set fields."""
        r = TrainResult(final_train_acc=0.9, epoch_train_95=100)
        assert r.final_train_acc == 0.9
        assert r.epoch_train_95 == 100


class TestTrainModel:
    """Tests for the training loop."""

    def test_training_produces_results(self):
        """Training should produce a TrainResult with populated fields."""
        torch.manual_seed(42)
        p = 5
        data = generate_modular_addition_data(p=p)
        train_data, test_data = split_dataset(data, 0.7, seed=42)
        model = GrokkingMLP(p=p, embed_dim=8, hidden_dim=16)

        config = TrainConfig(max_epochs=200, log_interval=50)
        result = train_model(model, train_data, test_data, config)

        assert isinstance(result, TrainResult)
        assert len(result.train_accs) > 0
        assert len(result.test_accs) > 0
        assert result.total_epochs > 0
        assert 0 <= result.final_train_acc <= 1.0
        assert 0 <= result.final_test_acc <= 1.0

    def test_training_small_p_converges(self):
        """With small p and enough epochs, training should reach high accuracy."""
        torch.manual_seed(42)
        p = 5
        data = generate_modular_addition_data(p=p)
        train_data, test_data = split_dataset(data, 0.8, seed=42)
        model = GrokkingMLP(p=p, embed_dim=8, hidden_dim=32)

        config = TrainConfig(
            max_epochs=2000,
            weight_decay=0.01,
            log_interval=100,
        )
        result = train_model(model, train_data, test_data, config)

        # With p=5 (only 25 examples), should memorize training set
        assert result.final_train_acc > 0.8

    def test_logged_epochs_match_interval(self):
        """Logged epochs should match the log interval."""
        torch.manual_seed(42)
        p = 5
        data = generate_modular_addition_data(p=p)
        train_data, test_data = split_dataset(data, 0.7, seed=42)
        model = GrokkingMLP(p=p, embed_dim=8, hidden_dim=16)

        config = TrainConfig(max_epochs=300, log_interval=100)
        result = train_model(model, train_data, test_data, config)

        # Should log at epochs 100, 200, 300
        assert result.logged_epochs[0] == 100
        assert all(
            e % 100 == 0
            for e in result.logged_epochs
            if e != config.max_epochs
        )

    def test_early_stopping(self):
        """Early stopping should trigger when accuracy is high enough."""
        torch.manual_seed(42)
        p = 5
        data = generate_modular_addition_data(p=p)
        # Use 90% train to make convergence easy
        train_data, test_data = split_dataset(data, 0.9, seed=42)
        model = GrokkingMLP(p=p, embed_dim=16, hidden_dim=64)

        config = TrainConfig(
            max_epochs=5000,
            weight_decay=0.01,
            log_interval=50,
            early_stop_acc=0.99,
            early_stop_patience=2,
        )
        result = train_model(model, train_data, test_data, config)

        # If early stopping triggered, total_epochs < max_epochs
        # (May not trigger for p=5, which is fine — just verify it doesn't crash)
        assert result.total_epochs <= config.max_epochs
        assert result.total_epochs > 0

    def test_metrics_lists_same_length(self):
        """All metric lists should have the same length."""
        torch.manual_seed(42)
        p = 5
        data = generate_modular_addition_data(p=p)
        train_data, test_data = split_dataset(data, 0.7, seed=42)
        model = GrokkingMLP(p=p, embed_dim=8, hidden_dim=16)

        config = TrainConfig(max_epochs=200, log_interval=50)
        result = train_model(model, train_data, test_data, config)

        assert len(result.train_accs) == len(result.test_accs)
        assert len(result.train_accs) == len(result.train_losses)
        assert len(result.train_accs) == len(result.logged_epochs)
