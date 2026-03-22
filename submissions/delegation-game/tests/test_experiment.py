"""Tests for the experiment runner."""

from src.experiment import build_configs, INCENTIVE_SCHEMES, WORKER_COMPOSITIONS, NOISE_LEVELS, SEEDS


class TestBuildConfigs:
    def test_correct_count(self):
        configs = build_configs()
        expected = (len(INCENTIVE_SCHEMES) * len(WORKER_COMPOSITIONS)
                    * len(NOISE_LEVELS) * len(SEEDS))
        assert len(configs) == expected
        assert len(configs) == 144

    def test_all_schemes_present(self):
        configs = build_configs()
        schemes = set(c.scheme_name for c in configs)
        assert schemes == set(INCENTIVE_SCHEMES)

    def test_all_seeds_present(self):
        configs = build_configs()
        seeds = set(c.seed for c in configs)
        assert seeds == set(SEEDS)

    def test_all_noise_levels_present(self):
        configs = build_configs()
        noises = set(c.noise_std for c in configs)
        assert noises == set(NOISE_LEVELS.values())

    def test_each_config_has_3_workers(self):
        configs = build_configs()
        for c in configs:
            assert len(c.worker_types) == 3
