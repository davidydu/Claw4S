# tests/test_experiment.py
import datetime
from src.experiment import ExperimentConfig, run_chart_analysis, build_chart_configs


def test_config_defaults():
    config = ExperimentConfig()
    assert config.start_year == 1984
    assert config.end_year == 2044


def test_build_configs():
    configs = build_chart_configs(start_year=2000, end_year=2001)
    # 1 year × 365 days × 12 时辰 = 4380
    assert len(configs) > 4000
    assert len(configs) < 5000


def test_run_single_chart():
    dt = datetime.datetime(2000, 6, 15, 10, 0)
    result = run_chart_analysis(dt)
    assert "bazi" in result
    assert "ziwei" in result
    assert "wuxing" in result
    for system in ["bazi", "ziwei", "wuxing"]:
        scores = result[system]["domain_scores"]
        assert set(scores.keys()) == {"career", "wealth", "relationships", "health", "overall"}


def test_run_chart_reproducible():
    dt = datetime.datetime(2000, 6, 15, 10, 0)
    r1 = run_chart_analysis(dt)
    r2 = run_chart_analysis(dt)
    assert r1["bazi"]["domain_scores"] == r2["bazi"]["domain_scores"]
