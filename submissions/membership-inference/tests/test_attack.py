"""Tests for membership inference attack pipeline."""

import numpy as np
from src.data import generate_gaussian_clusters, split_data, SEED
from src.models import create_and_train_model, get_predictions
from src.attack import (
    build_attack_dataset,
    train_attack_classifier,
    evaluate_attack,
    run_attack_for_width,
    _count_params,
)


def test_build_attack_dataset_shape():
    """Attack dataset has expected shape (balanced members/non-members)."""
    attack_X, attack_y = build_attack_dataset(
        hidden_width=32, n_shadow=2, seed=SEED
    )
    # 2 shadow models * 500 samples each (250 train + 250 test) = 1000
    assert attack_X.shape[0] == attack_y.shape[0]
    assert attack_X.shape[1] == 5  # n_classes
    # Check balanced
    n_members = (attack_y == 1).sum()
    n_nonmembers = (attack_y == 0).sum()
    assert n_members == n_nonmembers


def test_train_attack_classifier():
    """Attack classifier trains without error and produces predictions."""
    attack_X, attack_y = build_attack_dataset(
        hidden_width=32, n_shadow=2, seed=SEED
    )
    clf = train_attack_classifier(attack_X, attack_y, seed=SEED)
    probs = clf.predict_proba(attack_X[:10])
    assert probs.shape == (10, 2)


def test_evaluate_attack_returns_valid_metrics():
    """Attack evaluation returns AUC and accuracy in valid range."""
    X, y = generate_gaussian_clusters(n_samples=200, seed=SEED)
    X_tr, y_tr, X_te, y_te = split_data(X, y, seed=SEED)
    model, _ = create_and_train_model(
        hidden_width=32, X_train=X_tr, y_train=y_tr, seed=SEED
    )
    preds_train = get_predictions(model, X_tr)
    preds_test = get_predictions(model, X_te)

    attack_X, attack_y = build_attack_dataset(
        hidden_width=32, n_shadow=2, seed=SEED
    )
    clf = train_attack_classifier(attack_X, attack_y, seed=SEED)

    results = evaluate_attack(clf, preds_train, preds_test)
    assert 0.0 <= results["attack_auc"] <= 1.0
    assert 0.0 <= results["attack_accuracy"] <= 1.0


def test_run_attack_for_width_structure():
    """Full attack pipeline returns expected structure."""
    X, y = generate_gaussian_clusters(n_samples=200, seed=SEED)
    X_tr, y_tr, X_te, y_te = split_data(X, y, seed=SEED)

    result = run_attack_for_width(
        hidden_width=32,
        X_train=X_tr, y_train=y_tr,
        X_test=X_te, y_test=y_te,
        n_shadow=2, n_repeats=2, seed=SEED,
    )

    assert result["hidden_width"] == 32
    assert len(result["repeats"]) == 2
    assert 0.0 <= result["mean_attack_auc"] <= 1.0
    assert result["std_attack_auc"] >= 0.0
    assert "mean_overfit_gap" in result
    assert "n_params" in result


def test_count_params():
    """Parameter count formula matches expected values."""
    # w=16: (10*16+16) + (16*5+5) = 176 + 85 = 261
    assert _count_params(16) == 261
    # w=32: (10*32+32) + (32*5+5) = 352 + 165 = 517
    assert _count_params(32) == 517
    # w=256: (10*256+256) + (256*5+5) = 2816 + 1285 = 4101
    assert _count_params(256) == 4101
