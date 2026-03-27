"""Tests for adversarial example generation and transfer evaluation."""

import torch
from src.data import make_gaussian_clusters
from src.models import MLP
from src.train import train_model
from src.adversarial import fgsm_attack, evaluate_clean_accuracy, compute_transfer_rate


def _trained_model(width: int, depth: int = 2, seed: int = 42) -> tuple:
    """Helper: train a model and return (model, dataset)."""
    ds = make_gaussian_clusters(n_samples=100, n_features=10, n_classes=5, seed=seed)
    torch.manual_seed(seed)
    model = MLP(input_dim=10, n_classes=5, hidden_width=width, n_hidden_layers=depth)
    train_model(model, ds, lr=0.01, epochs=30, batch_size=32, seed=seed)
    return model, ds


def test_fgsm_produces_perturbation():
    """FGSM output differs from clean input."""
    model, ds = _trained_model(64)
    X, y = ds.tensors
    X_adv = fgsm_attack(model, X, y, epsilon=0.3)
    assert X_adv.shape == X.shape
    assert not torch.allclose(X_adv, X), "Adversarial examples should differ from clean"


def test_fgsm_perturbation_magnitude():
    """FGSM perturbation magnitude is bounded by epsilon."""
    model, ds = _trained_model(64)
    X, y = ds.tensors
    epsilon = 0.3
    X_adv = fgsm_attack(model, X, y, epsilon=epsilon)
    max_pert = (X_adv - X).abs().max().item()
    assert max_pert <= epsilon + 1e-6, f"Max perturbation {max_pert} > epsilon {epsilon}"


def test_clean_accuracy_above_chance():
    """Trained model achieves above-chance accuracy."""
    model, ds = _trained_model(128)
    X, y = ds.tensors
    acc = evaluate_clean_accuracy(model, X, y)
    assert acc > 0.3, f"Expected accuracy > 0.3 (chance=0.2), got {acc}"


def test_transfer_rate_in_bounds():
    """Transfer rate is in [0, 1]."""
    source, ds = _trained_model(64, seed=42)
    torch.manual_seed(43)
    target = MLP(input_dim=10, n_classes=5, hidden_width=128, n_hidden_layers=2)
    train_model(target, ds, lr=0.01, epochs=30, batch_size=32, seed=43)

    X, y = ds.tensors
    result = compute_transfer_rate(source, target, X, y, epsilon=0.3)
    assert 0.0 <= result["transfer_rate"] <= 1.0
    assert result["source_clean_acc"] > 0.2
    assert result["target_clean_acc"] > 0.2


def test_transfer_rate_keys():
    """compute_transfer_rate returns all expected keys."""
    source, ds = _trained_model(32, seed=42)
    torch.manual_seed(42)
    target = MLP(input_dim=10, n_classes=5, hidden_width=32, n_hidden_layers=2)
    train_model(target, ds, lr=0.01, epochs=30, batch_size=32, seed=42)

    X, y = ds.tensors
    result = compute_transfer_rate(source, target, X, y, epsilon=0.3)
    expected_keys = {
        "transfer_rate", "source_clean_acc", "target_clean_acc",
        "source_adv_acc", "target_adv_acc", "n_successful_source_advs",
    }
    assert set(result.keys()) == expected_keys


def test_self_transfer_is_high():
    """Transferring adversarial examples to the same model should have rate ~1.0."""
    model, ds = _trained_model(128, seed=42)
    X, y = ds.tensors
    result = compute_transfer_rate(model, model, X, y, epsilon=0.3)
    # Self-transfer: same model, so all successful adversarials fool it trivially
    assert result["transfer_rate"] >= 0.99, (
        f"Self-transfer should be ~1.0, got {result['transfer_rate']}"
    )
