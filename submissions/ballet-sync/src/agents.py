"""Dancer agent representation for Kuramoto simulation."""

from dataclasses import dataclass
import numpy as np


@dataclass
class DancerAgent:
    """A single dancer with phase, frequency, and stage position."""
    phase: float       # θ ∈ [0, 2π)
    frequency: float   # ω (natural frequency)
    x: float           # stage x position
    y: float           # stage y position


def create_dancers(n, omega0=1.0, sigma=0.5, stage_size=10.0, seed=0):
    """Create N dancers with random phases, Gaussian frequencies, grid positions."""
    rng = np.random.default_rng(seed)
    phases = rng.uniform(0, 2 * np.pi, n)
    frequencies = rng.normal(omega0, sigma, n)
    # Place on a grid within the stage
    side = int(np.ceil(np.sqrt(n)))
    spacing = stage_size / (side + 1)
    dancers = []
    for i in range(n):
        row, col = divmod(i, side)
        x = (col + 1) * spacing
        y = (row + 1) * spacing
        dancers.append(DancerAgent(
            phase=float(phases[i]),
            frequency=float(frequencies[i]),
            x=x, y=y,
        ))
    return dancers
