"""Dataset generators for sparsity experiments.

Two tasks:
1. Modular addition mod 97 (grokking-prone classification task)
2. Nonlinear regression (continuous prediction task)
"""

import torch


# Fixed prime modulus for modular arithmetic task
MODULUS = 97


def make_modular_addition_data(
    modulus: int = MODULUS,
    seed: int = 42,
    train_fraction: float = 0.3,
) -> dict:
    """Generate modular addition dataset: (a, b) -> (a + b) mod p.

    The inputs are one-hot encoded pairs, and the target is the class
    label (a + b) mod p. Using a small training fraction (0.3) creates
    the conditions where grokking can occur.

    Parameters
    ----------
    modulus : int
        The prime modulus p. Default 97.
    seed : int
        Random seed for train/test split.
    train_fraction : float
        Fraction of all (p*p) examples used for training.

    Returns
    -------
    dict
        Keys: 'x_train', 'y_train', 'x_test', 'y_test',
              'input_dim', 'output_dim', 'task_name'.
    """
    torch.manual_seed(seed)

    # All pairs (a, b) with a, b in [0, p-1]
    pairs = []
    labels = []
    for a in range(modulus):
        for b in range(modulus):
            pairs.append((a, b))
            labels.append((a + b) % modulus)

    n_total = len(pairs)
    n_train = int(n_total * train_fraction)

    # Shuffle indices deterministically
    perm = torch.randperm(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # One-hot encode: concatenate one-hot(a) and one-hot(b)
    input_dim = 2 * modulus

    def encode(indices):
        x_list = []
        y_list = []
        for i in indices:
            a, b = pairs[i]
            vec = torch.zeros(input_dim)
            vec[a] = 1.0
            vec[modulus + b] = 1.0
            x_list.append(vec)
            y_list.append(labels[i])
        return torch.stack(x_list), torch.tensor(y_list, dtype=torch.long)

    x_train, y_train = encode(train_idx)
    x_test, y_test = encode(test_idx)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "input_dim": input_dim,
        "output_dim": modulus,
        "task_name": f"modular_addition_mod{modulus}",
    }


def make_regression_data(
    n_train: int = 2000,
    n_test: int = 500,
    input_dim: int = 10,
    seed: int = 42,
) -> dict:
    """Generate a nonlinear regression dataset.

    Target: y = sin(x @ w1) + 0.5 * cos(x @ w2) + noise
    This creates a task where the network must learn nonlinear features
    but does not exhibit grokking.

    Parameters
    ----------
    n_train : int
        Number of training samples.
    n_test : int
        Number of test samples.
    input_dim : int
        Dimension of input features.
    seed : int
        Random seed.

    Returns
    -------
    dict
        Keys: 'x_train', 'y_train', 'x_test', 'y_test',
              'input_dim', 'output_dim', 'task_name'.
    """
    torch.manual_seed(seed)

    # Fixed random projections for the target function
    w1 = torch.randn(input_dim)
    w2 = torch.randn(input_dim)

    def generate(n):
        x = torch.randn(n, input_dim)
        proj1 = x @ w1
        proj2 = x @ w2
        y = torch.sin(proj1) + 0.5 * torch.cos(proj2)
        y = y + 0.05 * torch.randn(n)  # small noise
        return x, y.unsqueeze(1)

    x_train, y_train = generate(n_train)
    x_test, y_test = generate(n_test)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "input_dim": input_dim,
        "output_dim": 1,
        "task_name": "nonlinear_regression",
    }
