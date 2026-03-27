"""Tests for MLP model training and activation extraction."""

import torch
import numpy as np
import pytest

from src.data import generate_clean_data, make_datasets
from src.model import MLP, train_model, extract_activations


class TestMLP:
    """Tests for MLP architecture."""

    def test_forward_shape(self):
        model = MLP(input_dim=10, hidden_dim=64, n_classes=5)
        x = torch.randn(32, 10)
        out = model(x)
        assert out.shape == (32, 5)

    def test_penultimate_shape(self):
        model = MLP(input_dim=10, hidden_dim=64, n_classes=5)
        x = torch.randn(32, 10)
        h = model.get_penultimate(x)
        assert h.shape == (32, 64)

    def test_penultimate_nonnegative(self):
        """ReLU output should be non-negative."""
        model = MLP(input_dim=10, hidden_dim=64, n_classes=5)
        x = torch.randn(100, 10)
        h = model.get_penultimate(x)
        assert (h >= 0).all()


class TestTrainModel:
    """Tests for model training."""

    def test_trains_successfully(self):
        X, y = generate_clean_data(n_samples=100, n_features=10, n_classes=5)
        ds = make_datasets(X, y)
        model = train_model(ds, input_dim=10, hidden_dim=32, n_classes=5, epochs=10)
        assert isinstance(model, MLP)
        assert not model.training  # Should be in eval mode

    def test_achieves_nonrandom_accuracy(self):
        X, y = generate_clean_data(n_samples=200, n_features=10, n_classes=5, seed=42)
        ds = make_datasets(X, y)
        model = train_model(ds, input_dim=10, hidden_dim=64, n_classes=5, epochs=30)

        with torch.no_grad():
            preds = model(torch.from_numpy(X).float()).argmax(dim=1)
            acc = (preds == torch.from_numpy(y).long()).float().mean().item()
        # Should be better than random (20% for 5 classes)
        assert acc > 0.4


class TestExtractActivations:
    """Tests for activation extraction."""

    def test_output_shape(self):
        X, y = generate_clean_data(n_samples=50, n_features=10, n_classes=5)
        ds = make_datasets(X, y)
        model = train_model(ds, input_dim=10, hidden_dim=32, n_classes=5, epochs=5)
        acts = extract_activations(model, ds)
        assert acts.shape == (50, 32)

    def test_deterministic(self):
        X, y = generate_clean_data(n_samples=50, n_features=10, n_classes=5)
        ds = make_datasets(X, y)
        model = train_model(ds, input_dim=10, hidden_dim=32, n_classes=5, epochs=5, seed=42)
        a1 = extract_activations(model, ds)
        a2 = extract_activations(model, ds)
        torch.testing.assert_close(a1, a2)
