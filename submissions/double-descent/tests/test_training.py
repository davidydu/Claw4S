"""Tests for training routines."""

import torch

from src.data import generate_regression_data
from src.model import RandomFeaturesModel, create_mlp
from src.training import fit_random_features, train_mlp


class TestFitRandomFeatures:
    """Tests for fit_random_features."""

    def test_output_structure(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=50, n_test=20, d=5, seed=42
        )
        model = RandomFeaturesModel(5, 100, seed=42)
        result = fit_random_features(model, X_train, y_train, X_test, y_test)
        assert "train_loss" in result
        assert "test_loss" in result
        assert isinstance(result["train_loss"], float)
        assert isinstance(result["test_loss"], float)

    def test_overparameterized_low_train_loss(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=30, n_test=20, d=5, noise_std=0.1, seed=42
        )
        model = RandomFeaturesModel(5, 200, seed=42)
        result = fit_random_features(model, X_train, y_train, X_test, y_test)
        assert result["train_loss"] < 0.01

    def test_positive_losses(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=50, n_test=20, d=5, seed=42
        )
        model = RandomFeaturesModel(5, 30, seed=42)
        result = fit_random_features(model, X_train, y_train, X_test, y_test)
        assert result["train_loss"] >= 0
        assert result["test_loss"] >= 0


class TestTrainMlp:
    """Tests for train_mlp."""

    def test_loss_decreases(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=50, n_test=20, d=5, noise_std=0.1, seed=42
        )
        model = create_mlp(5, 20, seed=42)
        result = train_mlp(
            model, X_train, y_train, X_test, y_test,
            epochs=200, lr=0.001, record_every=50,
        )
        losses = result["train_loss_history"]
        assert losses[-1] < losses[0], "Training loss should decrease"

    def test_output_structure_no_recording(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=30, n_test=10, d=5, seed=42
        )
        model = create_mlp(5, 10, seed=42)
        result = train_mlp(
            model, X_train, y_train, X_test, y_test,
            epochs=50, lr=0.001, record_every=0,
        )
        assert "final_train_loss" in result
        assert "final_test_loss" in result
        assert isinstance(result["final_train_loss"], float)
        assert isinstance(result["final_test_loss"], float)

    def test_output_structure_with_recording(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=30, n_test=10, d=5, seed=42
        )
        model = create_mlp(5, 10, seed=42)
        result = train_mlp(
            model, X_train, y_train, X_test, y_test,
            epochs=100, lr=0.001, record_every=25,
        )
        assert "train_loss_history" in result
        assert "test_loss_history" in result
        assert "epoch_history" in result
        # Epochs 1, 25, 50, 75, 100 = 5 checkpoints
        assert len(result["epoch_history"]) == 5

    def test_losses_are_positive(self):
        X_train, y_train, X_test, y_test = generate_regression_data(
            n_train=30, n_test=10, d=5, seed=42
        )
        model = create_mlp(5, 10, seed=42)
        result = train_mlp(
            model, X_train, y_train, X_test, y_test,
            epochs=50, lr=0.001,
        )
        assert result["final_train_loss"] > 0
        assert result["final_test_loss"] > 0
