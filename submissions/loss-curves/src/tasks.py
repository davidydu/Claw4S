"""Task definitions: modular arithmetic and random-feature tasks."""

import warnings

import torch
import numpy as np


SEED = 42


def make_modular_addition_data(
    p: int = 97, seed: int = SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full modular addition dataset: (a, b) -> (a + b) mod p.

    Returns X of shape (p*p, 2) and y of shape (p*p,).
    """
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = (pairs[:, 0] + pairs[:, 1]) % p
    idx = rng.permutation(len(pairs))
    X = torch.tensor(pairs[idx], dtype=torch.long)
    y = torch.tensor(labels[idx], dtype=torch.long)
    return X, y


def make_modular_multiplication_data(
    p: int = 97, seed: int = SEED
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full modular multiplication dataset: (a, b) -> (a * b) mod p.

    Returns X of shape (p*p, 2) and y of shape (p*p,).
    """
    rng = np.random.default_rng(seed)
    pairs = np.array([(a, b) for a in range(p) for b in range(p)], dtype=np.int64)
    labels = (pairs[:, 0] * pairs[:, 1]) % p
    idx = rng.permutation(len(pairs))
    X = torch.tensor(pairs[idx], dtype=torch.long)
    y = torch.tensor(labels[idx], dtype=torch.long)
    return X, y


def make_regression_data(
    n_samples: int = 2000,
    input_dim: int = 20,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-feature regression: y = Xw + noise.

    Returns X of shape (n_samples, input_dim) and y of shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    X_np = np.asarray(rng.standard_normal((n_samples, input_dim)), dtype=np.float64)
    w = np.asarray(rng.standard_normal(input_dim), dtype=np.float64) / np.sqrt(input_dim)
    noise = np.asarray(0.1 * rng.standard_normal(n_samples), dtype=np.float64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        y_np = X_np @ w + noise
    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)


def make_classification_data(
    n_samples: int = 2000,
    input_dim: int = 20,
    n_classes: int = 5,
    seed: int = SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random-feature classification with n_classes.

    Returns X of shape (n_samples, input_dim) and y of shape (n_samples,).
    """
    rng = np.random.default_rng(seed)
    X_np = np.asarray(rng.standard_normal((n_samples, input_dim)), dtype=np.float64)
    W = np.asarray(rng.standard_normal((input_dim, n_classes)), dtype=np.float64) / np.sqrt(input_dim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        logits = X_np @ W
    y_np = logits.argmax(axis=1)
    return torch.tensor(X_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.long)


TASK_REGISTRY: dict[str, dict] = {
    "mod_add": {
        "make_data": make_modular_addition_data,
        "task_type": "classification",
        "n_classes": 97,
        "input_type": "embedding",
        "vocab_size": 97,
        "embed_dim": 16,
        "n_inputs": 2,
    },
    "mod_mul": {
        "make_data": make_modular_multiplication_data,
        "task_type": "classification",
        "n_classes": 97,
        "input_type": "embedding",
        "vocab_size": 97,
        "embed_dim": 16,
        "n_inputs": 2,
    },
    "regression": {
        "make_data": make_regression_data,
        "task_type": "regression",
        "input_type": "continuous",
        "input_dim": 20,
    },
    "classification": {
        "make_data": make_classification_data,
        "task_type": "classification",
        "n_classes": 5,
        "input_type": "continuous",
        "input_dim": 20,
    },
}
