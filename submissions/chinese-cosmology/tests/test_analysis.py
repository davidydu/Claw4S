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


def test_compute_statistics_includes_correlation_inference():
    records = []
    domains = ["career", "wealth", "relationships", "health", "overall"]

    for i in range(240):
        bazi = i / 239
        ziwei = np.clip(bazi + (0.02 if i % 2 == 0 else -0.02), 0, 1)
        wuxing = np.clip(1 - bazi, 0, 1)
        record = {"datetime": f"2000-01-01T{i % 24:02d}:00:00"}
        for domain in domains:
            record[f"bazi_{domain}"] = float(bazi)
            record[f"ziwei_{domain}"] = float(ziwei)
            record[f"wuxing_{domain}"] = float(wuxing)
        records.append(record)

    stats = compute_statistics(records)
    assert "correlation_inference" in stats

    career = stats["correlation_inference"]["career"]["bazi_ziwei"]
    assert set(career.keys()) == {
        "r",
        "ci_lower",
        "ci_upper",
        "p_value",
        "p_value_bonferroni",
        "n",
    }
    assert career["n"] == len(records)
    assert career["ci_lower"] <= career["r"] <= career["ci_upper"]
    assert 0.0 <= career["p_value"] <= 1.0
    assert 0.0 <= career["p_value_bonferroni"] <= 1.0
    assert career["p_value_bonferroni"] >= career["p_value"]
