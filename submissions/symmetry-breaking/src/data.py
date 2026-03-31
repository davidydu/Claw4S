"""Data generation for modular addition task (a + b mod P)."""

import torch
from typing import Tuple

# Fixed modulus for the modular addition task
MODULUS = 97


def generate_modular_addition_data(
    modulus: int = MODULUS,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate full dataset for (a + b) mod P.

    Each input is a one-hot encoding of (a, b) concatenated: [one_hot(a) | one_hot(b)].
    Label is (a + b) mod P.

    Returns train/test split (80/20).

    Args:
        modulus: The modulus P for modular addition.
        seed: Random seed for reproducible train/test split.

    Returns:
        x_train, y_train, x_test, y_test: Tensor pairs for training and testing.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Generate all P^2 pairs
    all_a = torch.arange(modulus)
    all_b = torch.arange(modulus)
    grid_a, grid_b = torch.meshgrid(all_a, all_b, indexing="ij")
    a_flat = grid_a.reshape(-1)
    b_flat = grid_b.reshape(-1)

    # One-hot encode inputs: [one_hot(a) | one_hot(b)]
    one_hot_a = torch.nn.functional.one_hot(a_flat, num_classes=modulus).float()
    one_hot_b = torch.nn.functional.one_hot(b_flat, num_classes=modulus).float()
    x = torch.cat([one_hot_a, one_hot_b], dim=1)  # shape: (P^2, 2*P)

    # Labels: (a + b) mod P
    y = (a_flat + b_flat) % modulus

    # Shuffle and split 80/20
    n = x.size(0)
    perm = torch.randperm(n, generator=gen)
    split = int(0.8 * n)

    x_train = x[perm[:split]]
    y_train = y[perm[:split]]
    x_test = x[perm[split:]]
    y_test = y[perm[split:]]

    return x_train, y_train, x_test, y_test
