"""Tests for membership inference attack."""

import numpy as np
import torch
from torch.utils.data import TensorDataset

from src.attack import (
    get_attack_features,
    train_attack_classifier,
    run_attack,
    AttackClassifier,
)
from src.model import MLP
from src.train import train_model


def _make_toy_data(n=50, d=5, c=3, seed=42):
    """Create toy data."""
    torch.manual_seed(seed)
    X = torch.randn(n, d)
    y = torch.randint(0, c, (n,))
    return TensorDataset(X, y)


def test_attack_features_shape():
    """Attack features have shape (n_samples, num_classes + 4)."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=16, num_classes=3)
    ds = _make_toy_data(n=20, d=5, c=3)
    features = get_attack_features(model, ds)
    # num_classes=3 + max_conf + entropy + loss + correct = 3 + 4 = 7
    assert features.shape == (20, 7)


def test_attack_features_softmax_sum_to_one():
    """First num_classes columns (softmax probs) sum to 1."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=16, num_classes=3)
    ds = _make_toy_data(n=20, d=5, c=3)
    features = get_attack_features(model, ds)
    # First 3 columns are softmax probabilities
    np.testing.assert_allclose(features[:, :3].sum(axis=1), 1.0, atol=1e-5)


def test_attack_features_max_conf_valid():
    """Max confidence is in (0, 1]."""
    torch.manual_seed(42)
    model = MLP(input_dim=5, hidden_dim=16, num_classes=3)
    ds = _make_toy_data(n=20, d=5, c=3)
    features = get_attack_features(model, ds)
    max_conf = features[:, 3]  # 4th column
    assert np.all(max_conf > 0)
    assert np.all(max_conf <= 1.0 + 1e-6)


def test_attack_classifier_forward():
    """Attack classifier produces scalar output."""
    clf = AttackClassifier(input_dim=7)
    x = torch.randn(8, 7)
    out = clf(x)
    assert out.shape == (8,)


def test_train_attack_classifier():
    """Attack classifier trains without error."""
    X = np.random.randn(100, 7).astype(np.float32)
    y = np.random.randint(0, 2, 100).astype(np.int64)
    clf = train_attack_classifier(X, y, epochs=5, seed=42)
    assert isinstance(clf, AttackClassifier)


def test_run_attack_returns_valid_metrics():
    """Attack returns AUC and accuracy in valid ranges."""
    torch.manual_seed(42)
    member_ds = _make_toy_data(n=30, d=5, c=3, seed=1)
    nonmember_ds = _make_toy_data(n=30, d=5, c=3, seed=2)

    model, _ = train_model(
        train_dataset=member_ds, input_dim=5, num_classes=3,
        epochs=10, batch_size=16, seed=42,
    )

    # Create attack classifier with correct feature dim (3 classes + 4 = 7)
    attack_X = np.random.randn(60, 7).astype(np.float32)
    attack_y = np.concatenate([np.ones(30), np.zeros(30)]).astype(np.int64)
    attack_clf = train_attack_classifier(attack_X, attack_y, epochs=10, seed=42)

    result = run_attack(attack_clf, model, member_ds, nonmember_ds)
    assert 0.0 <= result["attack_auc"] <= 1.0
    assert 0.0 <= result["attack_accuracy"] <= 1.0
    assert result["n_members"] == 30
    assert result["n_nonmembers"] == 30
