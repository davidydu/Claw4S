"""Tests for markdown report generation."""

from src.report import generate_report


def test_report_includes_ci_and_versions():
    """Report should include uncertainty and software version metadata."""
    all_results = {
        "metadata": {
            "seed": 42,
            "tasks": ["modular_arithmetic", "sine_regression"],
            "hidden_sizes": [64],
            "snapshot_epochs": [0, 100],
            "runtime_seconds": 12.3,
            "python_version": "3.13.5",
            "torch_version": "2.6.0",
            "numpy_version": "2.2.4",
            "scipy_version": "1.15.2",
            "matplotlib_version": "3.10.1",
        },
        "models": {
            "mod97_h64": {
                "0": {
                    "aggregate": {
                        "observed_dist": {
                            "1": 0.30,
                            "2": 0.18,
                            "3": 0.12,
                            "4": 0.10,
                            "5": 0.08,
                            "6": 0.07,
                            "7": 0.06,
                            "8": 0.05,
                            "9": 0.04,
                        },
                        "n_weights": 1000,
                        "chi2": 2.0,
                        "p_value": 0.98,
                        "mad": 0.010,
                        "mad_class": "acceptable",
                        "kl_div": 0.001,
                    },
                    "per_layer": {},
                },
                "100": {
                    "aggregate": {
                        "observed_dist": {
                            "1": 0.301,
                            "2": 0.176,
                            "3": 0.125,
                            "4": 0.097,
                            "5": 0.079,
                            "6": 0.066,
                            "7": 0.058,
                            "8": 0.053,
                            "9": 0.045,
                        },
                        "n_weights": 1000,
                        "chi2": 1.8,
                        "p_value": 0.99,
                        "mad": 0.009,
                        "mad_class": "acceptable",
                        "kl_div": 0.001,
                    },
                    "per_layer": {},
                },
            }
        },
        "controls": {
            "uniform": {
                "observed_dist": {
                    "1": 0.11,
                    "2": 0.11,
                    "3": 0.11,
                    "4": 0.11,
                    "5": 0.11,
                    "6": 0.11,
                    "7": 0.11,
                    "8": 0.11,
                    "9": 0.12,
                },
                "n_weights": 1000,
                "chi2": 300.0,
                "p_value": 0.0,
                "mad": 0.058,
                "mad_class": "nonconformity",
                "kl_div": 0.08,
            }
        },
    }

    report = generate_report(all_results)
    assert "95% CI" in report
    assert "Python 3.13.5" in report
