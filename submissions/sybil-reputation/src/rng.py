"""Deterministic RNG utilities used across the simulation and tests."""

from __future__ import annotations

import random
from typing import Sequence, Union


Population = Union[int, Sequence[int]]


class RNG:
    """Small wrapper matching the subset of NumPy RNG behavior we use."""

    def __init__(self, seed: int):
        self._rng = random.Random(seed)

    def uniform(self, low: float, high: float) -> float:
        return self._rng.uniform(low, high)

    def normal(self, mean: float, stddev: float) -> float:
        return self._rng.gauss(mean, stddev)

    def choice(self, population: Population, size: int = 1, replace: bool = True):
        if size < 1:
            return []

        if isinstance(population, int):
            values = list(range(population))
        else:
            values = list(population)

        if replace:
            picks = [self._rng.choice(values) for _ in range(size)]
        else:
            if size > len(values):
                raise ValueError("Cannot sample without replacement: size > population")
            picks = self._rng.sample(values, size)

        if size == 1:
            return picks[0]
        return picks


def default_rng(seed: int) -> RNG:
    """Construct a deterministic RNG instance from a seed."""
    return RNG(seed)
