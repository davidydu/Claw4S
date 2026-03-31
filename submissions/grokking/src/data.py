"""Modular arithmetic dataset generation for grokking experiments.

Generates all (a, b) pairs for modular addition mod p, and provides
deterministic train/test splitting.
"""

import numpy as np
import torch


def generate_modular_addition_data(p: int = 97) -> dict:
    """Generate all (a, b, (a + b) mod p) triples for modular addition.

    Args:
        p: The prime modulus. Default 97 (standard in grokking literature).

    Returns:
        Dictionary with keys:
            - "inputs": int64 tensor of shape (p*p, 2), each row is (a, b)
            - "labels": int64 tensor of shape (p*p,), each entry is (a+b) mod p
            - "p": the prime modulus
    """
    if p < 2:
        raise ValueError(f"Modulus p must be >= 2, got {p}")

    a_vals = np.arange(p)
    b_vals = np.arange(p)
    aa, bb = np.meshgrid(a_vals, b_vals, indexing="ij")
    aa = aa.flatten()
    bb = bb.flatten()
    labels = (aa + bb) % p

    inputs = torch.tensor(np.stack([aa, bb], axis=1), dtype=torch.long)
    labels = torch.tensor(labels, dtype=torch.long)

    return {"inputs": inputs, "labels": labels, "p": p}


def split_dataset(
    data: dict, train_fraction: float, seed: int = 42
) -> tuple[dict, dict]:
    """Split dataset into train and test sets deterministically.

    Args:
        data: Dataset dict from generate_modular_addition_data.
        train_fraction: Fraction of data to use for training (0 < frac < 1).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_data, test_data), each a dict with "inputs" and "labels".
    """
    if not 0 < train_fraction < 1:
        raise ValueError(
            f"train_fraction must be in (0, 1), got {train_fraction}"
        )

    n = len(data["inputs"])
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_fraction)
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    train_data = {
        "inputs": data["inputs"][train_idx],
        "labels": data["labels"][train_idx],
    }
    test_data = {
        "inputs": data["inputs"][test_idx],
        "labels": data["labels"][test_idx],
    }

    return train_data, test_data
