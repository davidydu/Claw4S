"""Tests for simulation runner."""

import numpy as np
import pytest

from src.simulation import run_simulation


def test_simulation_returns_expected_keys():
    result = run_simulation(
        composition=["open", "secretive", "reciprocal", "strategic"],
        competition=0.5,
        complementarity=0.5,
        n_rounds=100,
        seed=42,
    )
    assert "config" in result
    assert "summary" in result
    assert "time_series" in result


def test_simulation_config_echoed():
    result = run_simulation(
        composition=["open", "open", "open", "open"],
        competition=0.3,
        complementarity=0.7,
        n_rounds=50,
        seed=99,
    )
    c = result["config"]
    assert c["competition"] == 0.3
    assert c["complementarity"] == 0.7
    assert c["n_rounds"] == 50
    assert c["seed"] == 99


def test_all_open_sharing_rate():
    result = run_simulation(
        composition=["open", "open", "open", "open"],
        competition=0.5,
        complementarity=0.5,
        n_rounds=200,
        seed=42,
    )
    assert result["summary"]["tail_sharing_rate"] == pytest.approx(1.0)


def test_all_secretive_sharing_rate():
    result = run_simulation(
        composition=["secretive", "secretive", "secretive", "secretive"],
        competition=0.5,
        complementarity=0.5,
        n_rounds=200,
        seed=42,
    )
    assert result["summary"]["tail_sharing_rate"] == pytest.approx(0.0)


def test_reproducibility():
    r1 = run_simulation(["open", "strategic"], 0.5, 0.5, 100, seed=42)
    r2 = run_simulation(["open", "strategic"], 0.5, 0.5, 100, seed=42)
    assert r1["summary"]["avg_sharing_rate"] == r2["summary"]["avg_sharing_rate"]
    assert r1["summary"]["avg_group_welfare"] == r2["summary"]["avg_group_welfare"]
