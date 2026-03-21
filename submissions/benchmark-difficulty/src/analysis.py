"""Analysis module for benchmark difficulty prediction.

Computes correlations between structural features and IRT difficulty,
trains a Random Forest regressor, and performs cross-validation.

Key metrics:
  - Spearman rank correlation per feature
  - Random Forest R-squared and MAE
  - Cross-validated Spearman rho (primary metric)
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from src.data import load_arc_with_difficulty
from src.features import extract_all_features, FEATURE_NAMES


def compute_correlations(
    features_list: list[dict],
    difficulties: list[float],
) -> dict[str, tuple[float, float]]:
    """Compute Spearman rank correlations between features and difficulty.

    Args:
        features_list: List of feature dicts (one per question).
        difficulties: List of difficulty scores.

    Returns:
        Dict mapping feature name -> (rho, p_value).
    """
    diff_arr = np.array(difficulties)
    correlations = {}

    for name in FEATURE_NAMES:
        feat_arr = np.array([f[name] for f in features_list])
        # Handle constant features gracefully
        if np.std(feat_arr) < 1e-10:
            correlations[name] = (0.0, 1.0)
        else:
            rho, pval = stats.spearmanr(feat_arr, diff_arr)
            correlations[name] = (float(rho), float(pval))

    return correlations


def _features_to_matrix(features_list: list[dict]) -> np.ndarray:
    """Convert list of feature dicts to a numpy matrix."""
    return np.array([[f[name] for name in FEATURE_NAMES] for f in features_list])


def train_difficulty_model(
    features_list: list[dict],
    difficulties: list[float],
    seed: int = 42,
) -> dict:
    """Train a Random Forest regressor to predict difficulty from features.

    Args:
        features_list: List of feature dicts.
        difficulties: List of difficulty scores.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: model, r_squared, mae, feature_importances,
        predictions.
    """
    X = _features_to_matrix(features_list)
    y = np.array(difficulties)

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=3,
        random_state=seed,
    )
    model.fit(X, y)

    predictions = model.predict(X)
    residuals = y - predictions
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    mae = np.mean(np.abs(residuals))

    importances = dict(zip(FEATURE_NAMES, model.feature_importances_))

    return {
        "model": model,
        "r_squared": float(r_squared),
        "mae": float(mae),
        "feature_importances": importances,
        "predictions": predictions.tolist(),
    }


def cross_validate_model(
    features_list: list[dict],
    difficulties: list[float],
    n_folds: int = 5,
    seed: int = 42,
) -> dict:
    """Cross-validate the difficulty prediction model.

    Args:
        features_list: List of feature dicts.
        difficulties: List of difficulty scores.
        n_folds: Number of CV folds.
        seed: Random seed.

    Returns:
        Dict with keys: mean_r_squared, std_r_squared, mean_mae,
        mean_spearman, std_spearman, fold_scores.
    """
    X = _features_to_matrix(features_list)
    y = np.array(difficulties)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_scores = []
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=3,
            random_state=seed,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)
        residuals = y_test - predictions
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        mae = float(np.mean(np.abs(residuals)))

        # Spearman correlation on test set
        if len(y_test) > 2 and np.std(predictions) > 1e-10:
            rho, _ = stats.spearmanr(predictions, y_test)
        else:
            rho = 0.0

        fold_scores.append({
            "r_squared": float(r_squared),
            "mae": mae,
            "spearman_rho": float(rho),
        })

    r2_vals = [fs["r_squared"] for fs in fold_scores]
    mae_vals = [fs["mae"] for fs in fold_scores]
    rho_vals = [fs["spearman_rho"] for fs in fold_scores]

    return {
        "mean_r_squared": float(np.mean(r2_vals)),
        "std_r_squared": float(np.std(r2_vals)),
        "mean_mae": float(np.mean(mae_vals)),
        "std_mae": float(np.std(mae_vals)),
        "mean_spearman": float(np.mean(rho_vals)),
        "std_spearman": float(np.std(rho_vals)),
        "fold_scores": fold_scores,
    }


def run_full_analysis(
    use_hardcoded: bool = False,
    seed: int = 42,
) -> dict:
    """Run the complete analysis pipeline.

    Args:
        use_hardcoded: Whether to use hardcoded data only.
        seed: Random seed.

    Returns:
        Complete results dict with correlations, model metrics, CV metrics,
        feature importances, predictions, and difficulties.
    """
    # Load data
    questions = load_arc_with_difficulty(use_hardcoded=use_hardcoded)

    # Extract features
    features_list = extract_all_features(questions)
    difficulties = [q["difficulty"] for q in questions]

    # Compute correlations
    correlations = compute_correlations(features_list, difficulties)

    # Train model on full data
    model_result = train_difficulty_model(features_list, difficulties, seed=seed)

    # Cross-validate
    cv_result = cross_validate_model(features_list, difficulties, seed=seed)

    # Rank features by importance
    importances = model_result["feature_importances"]
    ranked_features = sorted(
        importances.items(), key=lambda x: x[1], reverse=True
    )

    return {
        "num_questions": len(questions),
        "correlations": {
            name: {"rho": rho, "pvalue": pval}
            for name, (rho, pval) in correlations.items()
        },
        "model_metrics": {
            "r_squared": model_result["r_squared"],
            "mae": model_result["mae"],
        },
        "cv_metrics": {
            "mean_r_squared": cv_result["mean_r_squared"],
            "std_r_squared": cv_result["std_r_squared"],
            "mean_mae": cv_result["mean_mae"],
            "std_mae": cv_result["std_mae"],
            "mean_spearman": cv_result["mean_spearman"],
            "std_spearman": cv_result["std_spearman"],
            "fold_scores": cv_result["fold_scores"],
        },
        "feature_importances": importances,
        "ranked_features": ranked_features,
        "predictions": model_result["predictions"],
        "difficulties": difficulties,
        "features_list": features_list,
        "question_ids": [q["id"] for q in questions],
        "seed": seed,
    }
