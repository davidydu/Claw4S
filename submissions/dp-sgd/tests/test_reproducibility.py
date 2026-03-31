"""Tests for reproducibility utilities."""

import numpy as np
import torch

from src.reproducibility import (
    configure_reproducibility,
    collect_reproducibility_metadata,
)


class TestConfigureReproducibility:
    """Tests for deterministic execution configuration."""

    def test_sets_deterministic_torch_flags(self):
        configure_reproducibility(seed=42)
        assert torch.are_deterministic_algorithms_enabled() is True
        assert torch.get_num_threads() == 1

    def test_seed_controls_numpy_and_torch_rng(self):
        configure_reproducibility(seed=123)
        np_val_1 = np.random.rand()
        torch_val_1 = torch.rand(1).item()

        configure_reproducibility(seed=123)
        np_val_2 = np.random.rand()
        torch_val_2 = torch.rand(1).item()

        assert np_val_1 == np_val_2
        assert torch_val_1 == torch_val_2


class TestCollectReproducibilityMetadata:
    """Tests for reproducibility metadata contract."""

    def test_contains_required_fields(self):
        configure_reproducibility(seed=42)
        meta = collect_reproducibility_metadata()

        expected_keys = {
            "python_version",
            "python_implementation",
            "torch_version",
            "numpy_version",
            "torch_deterministic_algorithms_enabled",
            "torch_num_threads",
        }
        assert expected_keys.issubset(set(meta))
        assert meta["torch_deterministic_algorithms_enabled"] is True
        assert meta["torch_num_threads"] >= 1
