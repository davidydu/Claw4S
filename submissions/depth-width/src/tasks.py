"""Task definitions: sparse parity (compositional) and smooth regression."""

import torch


def make_sparse_parity_data(
    n_bits: int = 20,
    k_relevant: int = 5,
    n_train: int = 2000,
    n_test: int = 1000,
    seed: int = 42,
) -> dict:
    """Generate a sparse parity classification task.

    The label is the XOR (parity) of k_relevant out of n_bits input bits.
    This is a well-studied task that provably requires depth for efficient
    learning: shallow networks need exponentially more width to learn parity.

    Reference: Barak et al. (2022) "Hidden Progress in Deep Learning"

    Args:
        n_bits: Total number of input bits.
        k_relevant: Number of bits whose parity determines the label.
        n_train: Number of training examples.
        n_test: Number of test examples.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: x_train, y_train, x_test, y_test, input_dim, output_dim.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    # Fix which bits are relevant (first k bits for reproducibility)
    relevant_bits = list(range(k_relevant))

    def generate_xy(n: int) -> tuple:
        # Random binary inputs in {-1, +1}
        x = 2.0 * torch.bernoulli(
            0.5 * torch.ones(n, n_bits), generator=gen
        ) - 1.0
        # Label = parity of relevant bits (product of {-1,+1} = parity)
        parity = torch.ones(n)
        for idx in relevant_bits:
            parity = parity * x[:, idx]
        # Map {-1, +1} -> {0, 1} for cross-entropy
        y = ((parity + 1) / 2).long()
        return x, y

    x_train, y_train = generate_xy(n_train)
    x_test, y_test = generate_xy(n_test)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "input_dim": n_bits,
        "output_dim": 2,  # binary classification
        "task_type": "classification",
        "task_name": "sparse_parity",
        "k_relevant": k_relevant,
    }


def make_regression_data(
    n_train: int = 2000,
    n_test: int = 500,
    input_dim: int = 8,
    seed: int = 42,
) -> dict:
    """Generate a smooth regression task: sum of sinusoidal features.

    Target: y = sum_i sin(x_i) + 0.1 * sum_{i<j} x_i * x_j
    This creates a smooth surface that wide networks can fit easily.

    Args:
        n_train: Number of training points.
        n_test: Number of test points.
        input_dim: Dimensionality of input features.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: x_train, y_train, x_test, y_test, input_dim, output_dim.
    """
    gen = torch.Generator()
    gen.manual_seed(seed)

    def generate_xy(n: int) -> tuple:
        x = torch.randn(n, input_dim, generator=gen)
        # Smooth target: sin components + pairwise interactions
        y = torch.sin(x).sum(dim=1)
        # Add pairwise interaction terms (scaled down)
        for i in range(input_dim):
            for j in range(i + 1, input_dim):
                y = y + 0.1 * x[:, i] * x[:, j]
        return x, y

    x_train, y_train = generate_xy(n_train)
    x_test, y_test = generate_xy(n_test)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "input_dim": input_dim,
        "output_dim": 1,
        "task_type": "regression",
        "task_name": "smooth_regression",
    }
