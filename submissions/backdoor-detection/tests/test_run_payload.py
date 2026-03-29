"""Tests for deterministic JSON payload generation in run.py."""

import run


def test_build_results_payload_excludes_timing_fields():
    raw_results = [
        {
            "config": {"poison_fraction": 0.1, "trigger_strength": 10.0, "hidden_dim": 64},
            "detection_auc": 0.95,
            "eigenvalue_ratio": 2.0,
            "top_5_eigenvalues": [2.0, 1.0],
            "clean_model_accuracy": 0.9,
            "backdoored_model_accuracy": 0.85,
            "backdoor_success_rate": 0.99,
            "n_poisoned": 50,
            "n_total": 500,
            "elapsed_seconds": 0.42,
        }
    ]

    payload = run.build_results_payload(
        raw_results,
        poison_fractions=[0.1],
        trigger_strengths=[10.0],
        hidden_dims=[64],
        seed=42,
    )

    assert "total_elapsed_seconds" not in payload["metadata"]
    assert "elapsed_seconds" not in payload["results"][0]
