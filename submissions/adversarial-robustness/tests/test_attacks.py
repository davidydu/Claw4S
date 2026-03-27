"""Tests for adversarial attack implementations."""

import torch

from src.data import make_dataloaders
from src.models import build_model
from src.train import train_model, evaluate_clean
from src.attacks import fgsm_attack, pgd_attack, evaluate_robust, EPSILONS


def _get_trained_model_and_data():
    """Helper: train a small model and return it with test data."""
    train_loader, _, X_test, y_test = make_dataloaders(
        dataset="circles", n_samples=500, seed=42)
    model = build_model(hidden_width=64, seed=42)
    train_model(model, train_loader, max_epochs=500, seed=42)
    return model, X_test, y_test


class TestFGSMAttack:
    def test_output_shape(self):
        model, X_test, y_test = _get_trained_model_and_data()
        X_adv = fgsm_attack(model, X_test, y_test, epsilon=0.1)
        assert X_adv.shape == X_test.shape

    def test_perturbation_bounded(self):
        model, X_test, y_test = _get_trained_model_and_data()
        eps = 0.1
        X_adv = fgsm_attack(model, X_test, y_test, epsilon=eps)
        max_perturbation = (X_adv - X_test).abs().max().item()
        assert max_perturbation <= eps + 1e-6, \
            f"Max perturbation {max_perturbation} exceeds epsilon {eps}"

    def test_zero_epsilon_no_change(self):
        model, X_test, y_test = _get_trained_model_and_data()
        X_adv = fgsm_attack(model, X_test, y_test, epsilon=0.0)
        assert torch.allclose(X_adv, X_test, atol=1e-6)

    def test_adversarial_reduces_accuracy(self):
        model, X_test, y_test = _get_trained_model_and_data()
        clean_acc = evaluate_clean(model, X_test, y_test)
        robust_acc = evaluate_robust(model, X_test, y_test, fgsm_attack, epsilon=0.5)
        assert robust_acc <= clean_acc + 0.01, \
            f"FGSM robust acc ({robust_acc}) should not exceed clean acc ({clean_acc})"


class TestPGDAttack:
    def test_output_shape(self):
        model, X_test, y_test = _get_trained_model_and_data()
        X_adv = pgd_attack(model, X_test, y_test, epsilon=0.1, n_steps=5)
        assert X_adv.shape == X_test.shape

    def test_perturbation_bounded(self):
        model, X_test, y_test = _get_trained_model_and_data()
        eps = 0.1
        X_adv = pgd_attack(model, X_test, y_test, epsilon=eps, n_steps=10)
        max_perturbation = (X_adv - X_test).abs().max().item()
        assert max_perturbation <= eps + 1e-6, \
            f"Max perturbation {max_perturbation} exceeds epsilon {eps}"

    def test_pgd_stronger_than_fgsm(self):
        model, X_test, y_test = _get_trained_model_and_data()
        fgsm_acc = evaluate_robust(model, X_test, y_test, fgsm_attack, epsilon=0.2)
        pgd_acc = evaluate_robust(model, X_test, y_test, pgd_attack, epsilon=0.2,
                                  n_steps=10)
        # PGD should find equal or stronger attacks
        assert pgd_acc <= fgsm_acc + 0.05, \
            f"PGD ({pgd_acc}) should be at most slightly weaker than FGSM ({fgsm_acc})"


class TestEpsilons:
    def test_epsilons_constant(self):
        assert EPSILONS == [0.01, 0.05, 0.1, 0.2, 0.5]

    def test_all_positive(self):
        assert all(e > 0 for e in EPSILONS)
