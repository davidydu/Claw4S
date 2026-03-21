"""Tests for analysis module."""

import numpy as np
import pytest
from src.analysis import (
    compute_correlations,
    train_difficulty_model,
    cross_validate_model,
    run_full_analysis,
)
from src.data import HARDCODED_ARC_SAMPLE
from src.features import extract_all_features, FEATURE_NAMES

# Seed for reproducibility
SEED = 42


@pytest.fixture
def sample_data():
    """Extract features from hardcoded ARC sample."""
    questions = HARDCODED_ARC_SAMPLE
    features_list = extract_all_features(questions)
    difficulties = [q["difficulty"] for q in questions]
    return features_list, difficulties


def test_compute_correlations_returns_dict(sample_data):
    """compute_correlations returns a dict of feature -> (rho, pvalue)."""
    features_list, difficulties = sample_data
    corrs = compute_correlations(features_list, difficulties)
    assert isinstance(corrs, dict)
    for name in FEATURE_NAMES:
        assert name in corrs, f"Missing correlation for {name}"
        rho, pval = corrs[name]
        assert -1.0 <= rho <= 1.0, f"Bad rho for {name}: {rho}"
        assert 0.0 <= pval <= 1.0, f"Bad pval for {name}: {pval}"


def test_train_difficulty_model(sample_data):
    """train_difficulty_model returns a dict with model and metrics."""
    features_list, difficulties = sample_data
    result = train_difficulty_model(features_list, difficulties, seed=SEED)
    assert "model" in result
    assert "r_squared" in result
    assert "mae" in result
    assert "feature_importances" in result
    assert isinstance(result["feature_importances"], dict)
    assert len(result["feature_importances"]) == len(FEATURE_NAMES)


def test_model_predictions_in_range(sample_data):
    """Model predictions should be roughly in [0, 1]."""
    features_list, difficulties = sample_data
    result = train_difficulty_model(features_list, difficulties, seed=SEED)
    model = result["model"]
    X = np.array([[f[name] for name in FEATURE_NAMES] for f in features_list])
    predictions = model.predict(X)
    # Allow small overshoot due to regression
    assert np.all(predictions >= -0.5), f"Predictions too low: {predictions.min()}"
    assert np.all(predictions <= 1.5), f"Predictions too high: {predictions.max()}"


def test_cross_validate_model(sample_data):
    """cross_validate_model returns CV scores."""
    features_list, difficulties = sample_data
    cv_result = cross_validate_model(
        features_list, difficulties, n_folds=3, seed=SEED
    )
    assert "mean_r_squared" in cv_result
    assert "std_r_squared" in cv_result
    assert "mean_mae" in cv_result
    assert "mean_spearman" in cv_result
    assert "fold_scores" in cv_result
    assert len(cv_result["fold_scores"]) == 3


def test_run_full_analysis():
    """run_full_analysis returns a complete results dict."""
    results = run_full_analysis(use_hardcoded=True, seed=SEED)
    assert "correlations" in results
    assert "model_metrics" in results
    assert "cv_metrics" in results
    assert "feature_importances" in results
    assert "num_questions" in results
    assert results["num_questions"] >= 50


def test_feature_importance_sum(sample_data):
    """Feature importances should sum to approximately 1."""
    features_list, difficulties = sample_data
    result = train_difficulty_model(features_list, difficulties, seed=SEED)
    total = sum(result["feature_importances"].values())
    assert abs(total - 1.0) < 0.01, f"Importance sum {total} not ~1.0"
