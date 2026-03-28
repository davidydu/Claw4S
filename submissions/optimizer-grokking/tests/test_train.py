"""Tests for training loop and optimizer creation."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import torch
from train import classify_training_history, make_optimizer, train_run
from model import ModularMLP
from data import split_data


def test_make_optimizer_sgd():
    """SGD optimizer should be created correctly."""
    model = ModularMLP(p=5)
    opt = make_optimizer(model, "sgd", lr=0.01, weight_decay=0.0)
    assert isinstance(opt, torch.optim.SGD)


def test_make_optimizer_sgd_momentum():
    """SGD with momentum should have momentum=0.9."""
    model = ModularMLP(p=5)
    opt = make_optimizer(model, "sgd_momentum", lr=0.01, weight_decay=0.0)
    assert isinstance(opt, torch.optim.SGD)
    assert opt.defaults["momentum"] == 0.9


def test_make_optimizer_adam():
    """Adam optimizer should be created correctly."""
    model = ModularMLP(p=5)
    opt = make_optimizer(model, "adam", lr=0.001, weight_decay=0.0)
    assert isinstance(opt, torch.optim.Adam)


def test_make_optimizer_adamw():
    """AdamW optimizer should be created correctly."""
    model = ModularMLP(p=5)
    opt = make_optimizer(model, "adamw", lr=0.001, weight_decay=0.01)
    assert isinstance(opt, torch.optim.AdamW)


def test_make_optimizer_unknown():
    """Unknown optimizer name should raise ValueError."""
    model = ModularMLP(p=5)
    with pytest.raises(ValueError, match="Unknown optimizer"):
        make_optimizer(model, "rmsprop", lr=0.01, weight_decay=0.0)


def test_train_run_short():
    """Short training run should return valid result dict."""
    train_ds, test_ds = split_data(p=5, seed=42)
    result = train_run(
        optimizer_name="adam",
        lr=0.01,
        weight_decay=0.0,
        train_a=train_ds.a,
        train_b=train_ds.b,
        train_t=train_ds.targets,
        test_a=test_ds.a,
        test_b=test_ds.b,
        test_t=test_ds.targets,
        max_epochs=10,
        batch_size=25,
        p=5,
        embed_dim=8,
        hidden_dim=16,
        seed=42,
        log_interval=5,
    )
    assert result["optimizer"] == "adam"
    assert result["lr"] == 0.01
    assert result["weight_decay"] == 0.0
    assert result["outcome"] in {
        "grokking",
        "direct_generalization",
        "memorization",
        "failure",
    }
    assert 0.0 <= result["final_train_acc"] <= 1.0
    assert 0.0 <= result["final_test_acc"] <= 1.0
    assert len(result["history"]) > 0


def test_train_run_history_structure():
    """History entries should have expected keys."""
    train_ds, test_ds = split_data(p=5, seed=42)
    result = train_run(
        optimizer_name="sgd",
        lr=0.01,
        weight_decay=0.0,
        train_a=train_ds.a,
        train_b=train_ds.b,
        train_t=train_ds.targets,
        test_a=test_ds.a,
        test_b=test_ds.b,
        test_t=test_ds.targets,
        max_epochs=5,
        batch_size=25,
        p=5,
        embed_dim=8,
        hidden_dim=16,
        seed=42,
        log_interval=1,
    )
    entry = result["history"][0]
    assert "epoch" in entry
    assert "train_acc" in entry
    assert "test_acc" in entry
    assert "train_loss" in entry
    assert "test_loss" in entry


def test_classify_training_history_same_logged_epoch_is_not_grokking():
    """Same-checkpoint threshold crossings should not count as delayed grokking."""
    history = [
        {"epoch": 1, "train_acc": 0.10, "test_acc": 0.10,
         "train_loss": 1.0, "test_loss": 1.0},
        {"epoch": 75, "train_acc": 0.96, "test_acc": 0.97,
         "train_loss": 0.1, "test_loss": 0.1},
    ]

    result = classify_training_history(history, acc_threshold=0.95)

    assert result["memorization_epoch"] == 75
    assert result["generalization_epoch"] == 75
    assert result["grokking_epoch"] is None
    assert result["outcome"] == "direct_generalization"


def test_train_run_all_optimizers():
    """All four optimizers should work without errors."""
    train_ds, test_ds = split_data(p=5, seed=42)
    for opt_name in ["sgd", "sgd_momentum", "adam", "adamw"]:
        result = train_run(
            optimizer_name=opt_name,
            lr=0.01,
            weight_decay=0.001,
            train_a=train_ds.a,
            train_b=train_ds.b,
            train_t=train_ds.targets,
            test_a=test_ds.a,
            test_b=test_ds.b,
            test_t=test_ds.targets,
            max_epochs=5,
            batch_size=25,
            p=5,
            embed_dim=8,
            hidden_dim=16,
            seed=42,
            log_interval=5,
        )
        assert result["optimizer"] == opt_name
        assert result["outcome"] in {
            "grokking",
            "direct_generalization",
            "memorization",
            "failure",
        }
