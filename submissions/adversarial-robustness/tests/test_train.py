"""Tests for training loop."""

import torch

from src.data import make_dataloaders
from src.models import build_model
from src.train import train_model, evaluate_clean


class TestTrainModel:
    def test_training_reduces_loss(self):
        train_loader, _, X_test, y_test = make_dataloaders(
            dataset="circles", n_samples=200, seed=42)
        model = build_model(hidden_width=32, seed=42)
        info = train_model(model, train_loader, max_epochs=100, seed=42)
        assert info["final_loss"] < info["loss_history"][0]

    def test_returns_expected_keys(self):
        train_loader, _, _, _ = make_dataloaders(
            dataset="circles", n_samples=200, seed=42)
        model = build_model(hidden_width=16, seed=42)
        info = train_model(model, train_loader, max_epochs=50, seed=42)
        assert "final_epoch" in info
        assert "final_loss" in info
        assert "loss_history" in info
        assert len(info["loss_history"]) == info["final_epoch"]

    def test_early_stopping(self):
        train_loader, _, _, _ = make_dataloaders(
            dataset="circles", n_samples=200, seed=42)
        model = build_model(hidden_width=64, seed=42)
        info = train_model(model, train_loader, max_epochs=2000,
                           patience=20, seed=42)
        # Should stop before max_epochs
        assert info["final_epoch"] < 2000


class TestEvaluateClean:
    def test_accuracy_range(self):
        train_loader, _, X_test, y_test = make_dataloaders(
            dataset="circles", n_samples=200, seed=42)
        model = build_model(hidden_width=32, seed=42)
        train_model(model, train_loader, max_epochs=200, seed=42)
        acc = evaluate_clean(model, X_test, y_test)
        assert 0.0 <= acc <= 1.0

    def test_trained_model_above_chance(self):
        train_loader, _, X_test, y_test = make_dataloaders(
            dataset="circles", n_samples=500, seed=42)
        model = build_model(hidden_width=64, seed=42)
        train_model(model, train_loader, max_epochs=500, seed=42)
        acc = evaluate_clean(model, X_test, y_test)
        assert acc > 0.6, f"Trained model accuracy {acc} should be above chance (0.5)"
