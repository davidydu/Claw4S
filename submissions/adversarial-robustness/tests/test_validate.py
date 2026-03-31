"""Tests for validation config inference."""

import validate


def test_infer_expected_config_uses_results_config():
    data = {
        "config": {
            "hidden_widths": [8, 16, 32],
            "epsilons": [0.01, 0.1],
            "seeds": [5, 7],
            "datasets": [{"name": "circles"}, {"name": "spirals"}],
        }
    }

    inferred = validate.infer_expected_config(data)
    assert inferred["widths"] == [8, 16, 32]
    assert inferred["epsilons"] == [0.01, 0.1]
    assert inferred["seeds"] == [5, 7]
    assert inferred["datasets"] == ["circles", "spirals"]
    assert inferred["n_results"] == 24


def test_infer_expected_config_falls_back_to_defaults():
    inferred = validate.infer_expected_config({})
    assert inferred["widths"] == validate.EXPECTED_WIDTHS
    assert inferred["epsilons"] == validate.EXPECTED_EPSILONS
    assert inferred["seeds"] == sorted(validate.EXPECTED_SEEDS)
    assert inferred["datasets"] == validate.EXPECTED_DATASETS
    assert inferred["n_results"] == 180
