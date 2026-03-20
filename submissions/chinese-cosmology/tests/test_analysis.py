# tests/test_analysis.py
import numpy as np
from src.analysis import (
    compute_null_model, bootstrap_ci, compute_statistics,
)


def test_null_model():
    """Null model should produce lower correlation than real data."""
    rng = np.random.default_rng(42)
    real = rng.uniform(0, 1, 100)
    correlated = real + rng.normal(0, 0.1, 100)
    null_scores = compute_null_model(real, np.clip(correlated, 0, 1),
                                      n_permutations=100, seed=42)
    assert len(null_scores) == 100
    assert np.mean(null_scores) < 0.5  # null should be low


def test_bootstrap_ci():
    values = [0.8, 0.82, 0.79, 0.81, 0.83]
    low, high = bootstrap_ci(values, confidence=0.95, n_bootstrap=1000, seed=42)
    assert low < high
    assert low > 0.7


def test_compute_statistics():
    records = [
        {"datetime": "2000-01-01T00:00", "bazi_career": 0.7, "ziwei_career": 0.65,
         "wuxing_career": 0.6, "bazi_wealth": 0.5, "ziwei_wealth": 0.55,
         "wuxing_wealth": 0.45}
        for _ in range(50)
    ]
    stats = compute_statistics(records)
    assert "correlation" in stats
    assert "domain_agreement" in stats
