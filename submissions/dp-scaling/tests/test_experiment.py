"""Tests for experiment orchestration outputs."""

import src.experiment as experiment


class TestRunFullExperiment:
    """Tests for run_full_experiment output structure."""

    def test_includes_environment_metadata_and_alpha_ci(self, monkeypatch):
        monkeypatch.setattr(experiment, "HIDDEN_SIZES", [16, 32, 64, 128, 256])
        monkeypatch.setattr(experiment, "SEEDS", [1, 2, 3])
        monkeypatch.setattr(
            experiment,
            "PRIVACY_CONFIGS",
            {
                "non_private": {"noise_multiplier": 0.0},
                "moderate_dp": {"noise_multiplier": 1.0},
            },
        )

        def fake_run_single(hidden_size, privacy_level, noise_multiplier, seed, **_kwargs):
            n_params = hidden_size * 20 + 5
            base_loss = 2.0 / (n_params ** 0.35)
            loss = base_loss * (1.0 + 0.2 * noise_multiplier) + seed * 1e-4
            return {
                "hidden_size": hidden_size,
                "n_params": n_params,
                "privacy_level": privacy_level,
                "noise_multiplier": noise_multiplier,
                "seed": seed,
                "test_loss": float(loss),
                "accuracy": 1.0,
                "train_time_s": 0.01,
            }

        monkeypatch.setattr(experiment, "run_single_experiment", fake_run_single)

        results = experiment.run_full_experiment()

        env = results["config"]["environment"]
        assert {"python_version", "torch_version", "numpy_version", "scipy_version"} <= set(
            env.keys()
        )

        for level, fit in results["scaling_fits"].items():
            assert "alpha_ci95" in fit
            assert len(fit["alpha_ci95"]) == 2
            assert fit["alpha_ci95"][0] <= fit["alpha"] <= fit["alpha_ci95"][1]

            summary_entry = results["summary"][level]
            assert "alpha_ci95" in summary_entry
            assert summary_entry["alpha_ci95"] == fit["alpha_ci95"]
