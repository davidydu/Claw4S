"""Training loop with grokking detection.

Trains a ModularMLP and records per-epoch train/test accuracy and loss.
Detects grokking: train_acc > 95% first, then test_acc > 95% later.
Uses pre-shuffled mini-batches (no DataLoader overhead) for speed.
"""

import torch
import torch.nn as nn

from model import ModularMLP

SEED = 42


def make_optimizer(
    model: nn.Module,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create an optimizer by name.

    Args:
        model: The model whose parameters to optimize.
        optimizer_name: One of "sgd", "sgd_momentum", "adam", "adamw".
        lr: Learning rate.
        weight_decay: Weight decay coefficient.

    Returns:
        A configured optimizer instance.

    Raises:
        ValueError: If optimizer_name is not recognized.
    """
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name!r}. "
                         f"Expected one of: sgd, sgd_momentum, adam, adamw")


def train_run(
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    train_a: torch.Tensor,
    train_b: torch.Tensor,
    train_t: torch.Tensor,
    test_a: torch.Tensor,
    test_b: torch.Tensor,
    test_t: torch.Tensor,
    max_epochs: int = 3000,
    batch_size: int = 512,
    p: int = 97,
    embed_dim: int = 32,
    hidden_dim: int = 64,
    seed: int = SEED,
    log_interval: int = 50,
) -> dict:
    """Train a single model with mini-batches and detect grokking.

    Uses pre-allocated tensors instead of DataLoader for speed.

    Args:
        optimizer_name: One of "sgd", "sgd_momentum", "adam", "adamw".
        lr: Learning rate.
        weight_decay: Weight decay coefficient.
        train_a, train_b, train_t: Training data tensors.
        test_a, test_b, test_t: Test data tensors.
        max_epochs: Maximum number of training epochs.
        batch_size: Mini-batch size for training.
        p: Modular arithmetic prime.
        embed_dim: Embedding dimension.
        hidden_dim: Hidden layer dimension.
        seed: Random seed for model initialization.
        log_interval: How often to record metrics (every N epochs).

    Returns:
        Dictionary with keys:
            optimizer, lr, weight_decay, history (list of dicts),
            final_train_acc, final_test_acc, memorization_epoch,
            grokking_epoch, outcome ("grokking", "memorization", "failure").
    """
    torch.manual_seed(seed)

    model = ModularMLP(p=p, embed_dim=embed_dim, hidden_dim=hidden_dim)
    optimizer = make_optimizer(model, optimizer_name, lr, weight_decay)
    criterion = nn.CrossEntropyLoss()

    n_train = train_a.size(0)
    history = []
    memorization_epoch = None
    grokking_epoch = None
    acc_threshold = 0.95

    # Pre-generate shuffle indices for reproducibility
    gen = torch.Generator()
    gen.manual_seed(seed)

    for epoch in range(1, max_epochs + 1):
        # Shuffle training data
        perm = torch.randperm(n_train, generator=gen)
        a_shuf = train_a[perm]
        b_shuf = train_b[perm]
        t_shuf = train_t[perm]

        # Mini-batch training
        model.train()
        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            a_batch = a_shuf[start:end]
            b_batch = b_shuf[start:end]
            t_batch = t_shuf[start:end]

            optimizer.zero_grad()
            logits = model(a_batch, b_batch)
            loss = criterion(logits, t_batch)
            loss.backward()
            optimizer.step()

        # Evaluate periodically
        if epoch % log_interval == 0 or epoch == 1 or epoch == max_epochs:
            model.eval()
            with torch.no_grad():
                train_logits = model(train_a, train_b)
                train_acc = (train_logits.argmax(-1) == train_t).float().mean().item()
                train_loss = criterion(train_logits, train_t).item()

                test_logits = model(test_a, test_b)
                test_acc = (test_logits.argmax(-1) == test_t).float().mean().item()
                test_loss = criterion(test_logits, test_t).item()

            history.append({
                "epoch": epoch,
                "train_acc": train_acc,
                "test_acc": test_acc,
                "train_loss": train_loss,
                "test_loss": test_loss,
            })

            if memorization_epoch is None and train_acc > acc_threshold:
                memorization_epoch = epoch

            if (memorization_epoch is not None
                    and grokking_epoch is None
                    and test_acc > acc_threshold):
                grokking_epoch = epoch

    final_train_acc = history[-1]["train_acc"] if history else 0.0
    final_test_acc = history[-1]["test_acc"] if history else 0.0

    # Classify outcome
    if grokking_epoch is not None:
        outcome = "grokking"
    elif memorization_epoch is not None:
        outcome = "memorization"
    else:
        outcome = "failure"

    return {
        "optimizer": optimizer_name,
        "lr": lr,
        "weight_decay": weight_decay,
        "history": history,
        "final_train_acc": final_train_acc,
        "final_test_acc": final_test_acc,
        "memorization_epoch": memorization_epoch,
        "grokking_epoch": grokking_epoch,
        "outcome": outcome,
    }
