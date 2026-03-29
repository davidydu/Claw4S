"""Utilities for deterministic execution and reproducibility metadata."""

import platform
from typing import Any

import numpy as np
import torch


def configure_reproducibility(seed: int | None = None) -> None:
    """Configure deterministic execution settings for NumPy and torch.

    Args:
        seed: Optional RNG seed. If provided, seeds both NumPy and torch.
    """
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(1)


def collect_reproducibility_metadata() -> dict[str, Any]:
    """Return runtime metadata required for reproducibility audits."""
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "torch_deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "torch_num_threads": torch.get_num_threads(),
    }
