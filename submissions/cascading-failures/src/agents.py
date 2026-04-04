"""Agent types for cascade simulations.

Each agent processes neighbor outputs through a noisy function and produces
its own output.  Three types model different error-handling strategies:

  - Robust:    clips extreme inputs to [-CLIP, CLIP], takes mean, applies tanh
  - Fragile:   takes mean of neighbors with NO nonlinearity (linear relay)
  - Averaging: takes mean of neighbor outputs, applies tanh (dampens via saturation)

The key difference: fragile agents relay the mean linearly (errors propagate
at full strength), while robust agents clip first AND apply tanh, and averaging
agents apply tanh (which saturates large errors). This ensures fragile agents
are genuinely the most vulnerable to cascades.
"""

from __future__ import annotations

import math
from typing import List

# Clipping bound for robust agents
CLIP_BOUND = 2.0

# Processing noise standard deviation (small)
NOISE_STD = 0.01

# Decay factor: output = decay * f(inputs) + noise
# Keeps clean-state outputs near 0 without requiring shock to observe drift
DECAY = 0.95


def _tanh_process(value: float, noise: float) -> float:
    """Nonlinear processing: tanh squashes large values, dampening errors."""
    return DECAY * math.tanh(value) + noise


def robust_agent(neighbor_outputs: List[float], noise: float) -> float:
    """Clips inputs to [-CLIP_BOUND, CLIP_BOUND], takes mean, applies tanh.

    Double protection: clipping prevents extreme values from entering,
    and tanh squashes whatever gets through.
    """
    if not neighbor_outputs:
        return noise
    clipped = [max(-CLIP_BOUND, min(CLIP_BOUND, x)) for x in neighbor_outputs]
    mean_val = sum(clipped) / len(clipped)
    return _tanh_process(mean_val, noise)


def fragile_agent(neighbor_outputs: List[float], noise: float) -> float:
    """Takes mean of neighbors and relays linearly (no nonlinearity).

    This is the most vulnerable agent: errors propagate at full strength
    through the mean, with only mild decay preventing unbounded growth.
    """
    if not neighbor_outputs:
        return noise
    mean_val = sum(neighbor_outputs) / len(neighbor_outputs)
    return DECAY * mean_val + noise


def averaging_agent(neighbor_outputs: List[float], noise: float) -> float:
    """Takes mean of neighbor outputs, applies tanh (dampens via saturation).

    The mean reduces outlier influence, and tanh squashes large values.
    More resilient than fragile but less protected than robust.
    """
    if not neighbor_outputs:
        return noise
    mean_val = sum(neighbor_outputs) / len(neighbor_outputs)
    return _tanh_process(mean_val, noise)


AGENT_TYPES = {
    "robust": robust_agent,
    "fragile": fragile_agent,
    "averaging": averaging_agent,
}
