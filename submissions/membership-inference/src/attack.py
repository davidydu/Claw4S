"""Membership inference attack using the shadow model approach.

Implements the Shokri et al. (2017) shadow model attack:
1. Train shadow models with the same architecture as the target
2. Use shadow models' train/test predictions to build attack training data
3. Train a logistic regression attack classifier on prediction confidence vectors
4. Evaluate attack on target model's train (member) vs test (non-member) data
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from typing import Dict, List, Tuple

from src.models import (
    create_and_train_model,
    get_predictions,
    compute_accuracy,
    SEED,
)
from src.data import generate_shadow_data, split_data


N_SHADOW_MODELS = 3


def build_attack_dataset(
    hidden_width: int,
    n_shadow: int = N_SHADOW_MODELS,
    input_dim: int = 10,
    output_dim: int = 5,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build attack training data from shadow models.

    For each shadow model, predictions on its training set are labeled
    as "member" (1) and predictions on its test set are labeled as
    "non-member" (0).

    Args:
        hidden_width: Hidden layer width (same architecture as target).
        n_shadow: Number of shadow models to train.
        input_dim: Input feature dimension.
        output_dim: Number of output classes.
        seed: Base random seed.

    Returns:
        attack_X: Prediction confidence vectors, shape (n, output_dim).
        attack_y: Membership labels (1=member, 0=non-member), shape (n,).
    """
    attack_X_parts: List[np.ndarray] = []
    attack_y_parts: List[np.ndarray] = []

    for shadow_idx in range(n_shadow):
        # Generate independent shadow data
        X_shadow, y_shadow = generate_shadow_data(
            shadow_idx=shadow_idx, seed=seed
        )

        # Split into train/test
        shadow_split_seed = seed + 2000 + shadow_idx * 50
        X_s_train, y_s_train, X_s_test, y_s_test = split_data(
            X_shadow, y_shadow, train_fraction=0.5, seed=shadow_split_seed
        )

        # Train shadow model
        shadow_train_seed = seed + 3000 + shadow_idx * 50
        shadow_model, _ = create_and_train_model(
            hidden_width=hidden_width,
            X_train=X_s_train,
            y_train=y_s_train,
            input_dim=input_dim,
            output_dim=output_dim,
            seed=shadow_train_seed,
        )

        # Get predictions on train (members) and test (non-members)
        preds_train = get_predictions(shadow_model, X_s_train)
        preds_test = get_predictions(shadow_model, X_s_test)

        attack_X_parts.append(preds_train)
        attack_X_parts.append(preds_test)
        attack_y_parts.append(np.ones(len(preds_train), dtype=np.int64))
        attack_y_parts.append(np.zeros(len(preds_test), dtype=np.int64))

    attack_X = np.concatenate(attack_X_parts, axis=0)
    attack_y = np.concatenate(attack_y_parts, axis=0)

    return attack_X, attack_y


def train_attack_classifier(
    attack_X: np.ndarray,
    attack_y: np.ndarray,
    seed: int = SEED,
) -> LogisticRegression:
    """Train a logistic regression attack classifier.

    Args:
        attack_X: Prediction confidence features.
        attack_y: Membership labels.
        seed: Random seed.

    Returns:
        Trained LogisticRegression classifier.
    """
    clf = LogisticRegression(
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    clf.fit(attack_X, attack_y)
    return clf


def evaluate_attack(
    attack_clf: LogisticRegression,
    target_preds_train: np.ndarray,
    target_preds_test: np.ndarray,
) -> Dict[str, float]:
    """Evaluate membership inference attack on target model.

    Args:
        attack_clf: Trained attack classifier.
        target_preds_train: Target model predictions on its training data (members).
        target_preds_test: Target model predictions on its test data (non-members).

    Returns:
        Dictionary with attack_auc and attack_accuracy.
    """
    # Combine member and non-member predictions
    X_eval = np.concatenate([target_preds_train, target_preds_test], axis=0)
    y_eval = np.concatenate([
        np.ones(len(target_preds_train), dtype=np.int64),
        np.zeros(len(target_preds_test), dtype=np.int64),
    ])

    # Get attack predictions
    attack_probs = attack_clf.predict_proba(X_eval)[:, 1]
    attack_preds = attack_clf.predict(X_eval)

    auc = float(roc_auc_score(y_eval, attack_probs))
    accuracy = float(accuracy_score(y_eval, attack_preds))

    return {
        "attack_auc": auc,
        "attack_accuracy": accuracy,
    }


def run_attack_for_width(
    hidden_width: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_shadow: int = N_SHADOW_MODELS,
    n_repeats: int = 3,
    seed: int = SEED,
) -> Dict[str, object]:
    """Run the full membership inference attack pipeline for one model width.

    Trains target models across multiple repeats, builds shadow-model-based
    attack classifiers, and evaluates attack success.

    Args:
        hidden_width: Hidden layer width.
        X_train: Target training features.
        y_train: Target training labels.
        X_test: Target test features.
        y_test: Target test labels.
        n_shadow: Number of shadow models per repeat.
        n_repeats: Number of independent repeats for variance estimation.
        seed: Base random seed.

    Returns:
        Dictionary with per-repeat and aggregated results.
    """
    repeat_results = []

    for rep in range(n_repeats):
        rep_seed = seed + rep * 500

        # Train target model
        target_model, _ = create_and_train_model(
            hidden_width=hidden_width,
            X_train=X_train,
            y_train=y_train,
            seed=rep_seed,
        )

        # Compute train/test accuracy and overfitting gap
        train_acc = compute_accuracy(target_model, X_train, y_train)
        test_acc = compute_accuracy(target_model, X_test, y_test)
        overfit_gap = train_acc - test_acc

        # Get target model predictions for attack evaluation
        target_preds_train = get_predictions(target_model, X_train)
        target_preds_test = get_predictions(target_model, X_test)

        # Build attack dataset from shadow models
        attack_X, attack_y = build_attack_dataset(
            hidden_width=hidden_width,
            n_shadow=n_shadow,
            seed=rep_seed,
        )

        # Train and evaluate attack classifier
        attack_clf = train_attack_classifier(attack_X, attack_y, seed=rep_seed)
        attack_results = evaluate_attack(
            attack_clf, target_preds_train, target_preds_test
        )

        repeat_results.append({
            "repeat": rep,
            "hidden_width": hidden_width,
            "train_acc": train_acc,
            "test_acc": test_acc,
            "overfit_gap": overfit_gap,
            "attack_auc": attack_results["attack_auc"],
            "attack_accuracy": attack_results["attack_accuracy"],
        })

    # Aggregate across repeats
    auc_values = [r["attack_auc"] for r in repeat_results]
    acc_values = [r["attack_accuracy"] for r in repeat_results]
    gap_values = [r["overfit_gap"] for r in repeat_results]
    train_acc_values = [r["train_acc"] for r in repeat_results]
    test_acc_values = [r["test_acc"] for r in repeat_results]

    return {
        "hidden_width": hidden_width,
        "n_params": _count_params(hidden_width),
        "mean_attack_auc": float(np.mean(auc_values)),
        "std_attack_auc": float(np.std(auc_values)),
        "mean_attack_accuracy": float(np.mean(acc_values)),
        "std_attack_accuracy": float(np.std(acc_values)),
        "mean_overfit_gap": float(np.mean(gap_values)),
        "std_overfit_gap": float(np.std(gap_values)),
        "mean_train_acc": float(np.mean(train_acc_values)),
        "mean_test_acc": float(np.mean(test_acc_values)),
        "repeats": repeat_results,
    }


def _count_params(hidden_width: int, input_dim: int = 10, output_dim: int = 5) -> int:
    """Count total parameters in a 2-layer MLP.

    Layer 1: input_dim * hidden_width + hidden_width (weights + bias)
    Layer 2: hidden_width * output_dim + output_dim (weights + bias)
    """
    return (input_dim * hidden_width + hidden_width) + (
        hidden_width * output_dim + output_dim
    )
