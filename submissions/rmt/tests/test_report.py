"""Tests for markdown report generation."""

from src.report import generate_report


def sample_results_data() -> dict:
    """Build a compact results payload for report tests."""
    return {
        "metadata": {
            "seed": 42,
            "hidden_dims": [32],
            "mod_epochs": 500,
            "reg_epochs": 500,
            "learning_rate": 1e-3,
            "torch_version": "2.6.0",
            "numpy_version": "2.2.4",
        },
        "training_results": [
            {
                "model_label": "mod97_h32",
                "task": "mod97",
                "hidden_dim": 32,
                "final_loss": 0.1,
                "final_accuracy": 0.9,
            },
            {
                "model_label": "regression_h32",
                "task": "regression",
                "hidden_dim": 32,
                "final_loss": 0.0,
                "final_mse": 0.0001,
            },
        ],
        "trained_analysis": [
            {
                "model_label": "mod97_h32",
                "layer_name": "fc1",
                "shape": [194, 32],
                "ks_statistic": 0.40,
                "outlier_fraction": 0.20,
                "spectral_norm_ratio": 2.50,
                "kl_divergence": 1.20,
                "n_eigenvalues": 32,
            },
            {
                "model_label": "regression_h32",
                "layer_name": "fc3",
                "shape": [32, 1],
                "ks_statistic": 0.00,
                "outlier_fraction": 0.00,
                "spectral_norm_ratio": 0.00,
                "kl_divergence": 0.00,
                "n_eigenvalues": 1,
            },
        ],
        "untrained_analysis": [
            {
                "model_label": "mod97_h32",
                "layer_name": "fc1",
                "shape": [194, 32],
                "ks_statistic": 0.05,
                "outlier_fraction": 0.00,
                "spectral_norm_ratio": 0.90,
                "kl_divergence": 0.05,
                "n_eigenvalues": 32,
            },
            {
                "model_label": "regression_h32",
                "layer_name": "fc3",
                "shape": [32, 1],
                "ks_statistic": 0.00,
                "outlier_fraction": 0.00,
                "spectral_norm_ratio": 0.00,
                "kl_divergence": 0.00,
                "n_eigenvalues": 1,
            },
        ],
        "delta_ks_summary": {
            "n_pairs": 2,
            "n_positive": 1,
            "n_negative": 0,
            "n_ties": 1,
            "positive_fraction": 0.5,
            "avg_delta": 0.175,
            "median_delta": 0.175,
            "std_delta": 0.2475,
            "sign_test_pvalue": 0.5,
            "bootstrap_ci_low": 0.0,
            "bootstrap_ci_high": 0.35,
        },
    }


def sample_all_positive_results_data() -> dict:
    """Build a payload where every layer increases after training."""
    return {
        "metadata": {
            "seed": 42,
            "hidden_dims": [32],
            "mod_epochs": 500,
            "reg_epochs": 500,
            "learning_rate": 1e-3,
            "torch_version": "2.6.0",
            "numpy_version": "2.2.4",
        },
        "training_results": [
            {
                "model_label": "mod97_h32",
                "task": "mod97",
                "hidden_dim": 32,
                "final_loss": 0.1,
                "final_accuracy": 0.9,
            },
        ],
        "trained_analysis": [
            {
                "model_label": "mod97_h32",
                "layer_name": "fc1",
                "shape": [194, 32],
                "ks_statistic": 0.40,
                "outlier_fraction": 0.20,
                "spectral_norm_ratio": 2.50,
                "kl_divergence": 1.20,
                "n_eigenvalues": 32,
            },
        ],
        "untrained_analysis": [
            {
                "model_label": "mod97_h32",
                "layer_name": "fc1",
                "shape": [194, 32],
                "ks_statistic": 0.05,
                "outlier_fraction": 0.00,
                "spectral_norm_ratio": 0.90,
                "kl_divergence": 0.05,
                "n_eigenvalues": 32,
            },
        ],
        "delta_ks_summary": {
            "n_pairs": 1,
            "n_positive": 1,
            "n_negative": 0,
            "n_ties": 0,
            "positive_fraction": 1.0,
            "avg_delta": 0.35,
            "median_delta": 0.35,
            "std_delta": 0.0,
            "sign_test_pvalue": 0.5,
            "bootstrap_ci_low": 0.35,
            "bootstrap_ci_high": 0.35,
        },
    }


def test_report_omits_wall_clock_timestamp():
    """Report text should stay deterministic across identical reruns."""
    report = generate_report(sample_results_data())

    assert "Generated:" not in report
    assert "Seed: 42" in report


def test_report_calls_out_degenerate_non_increasing_layers():
    """Non-increasing delta layers should be explained when degenerate."""
    report = generate_report(sample_results_data())

    assert "Layers with increased deviation:** 1/2" in report
    assert (
        "Layers without increased deviation:** 1/2, all single-eigenvalue "
        "layers where MP comparisons are degenerate."
    ) in report


def test_report_keeps_numbering_compact_when_all_layers_increase():
    """Key finding numbering should stay contiguous when no caveat line is needed."""
    report = generate_report(sample_all_positive_results_data())

    assert "Layers without increased deviation:" not in report
    assert "5. **Avg spectral norm ratio:**" in report


def test_report_includes_significance_and_confidence_interval():
    """Report should include paired-sign and CI summary for delta KS."""
    report = generate_report(sample_results_data())

    assert "Sign test p-value (trained KS > untrained KS):" in report
    assert "95% bootstrap CI for mean delta KS:" in report
