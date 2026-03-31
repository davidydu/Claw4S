"""Training loop for symmetry-breaking experiments."""

import torch
import torch.nn as nn
from typing import Dict, List, Any

from src.model import SymmetricMLP
from src.data import generate_modular_addition_data, MODULUS
from src.metrics import symmetry_metric


def train_single_run(
    hidden_dim: int,
    epsilon: float,
    num_epochs: int = 2000,
    batch_size: int = 256,
    lr: float = 0.1,
    log_interval: int = 50,
    seed: int = 42,
    modulus: int = MODULUS,
) -> Dict[str, Any]:
    """Train one model and record symmetry trajectory.

    Args:
        hidden_dim: Number of hidden neurons.
        epsilon: Perturbation scale for symmetric init.
        num_epochs: Number of training epochs.
        batch_size: Mini-batch size for SGD.
        lr: Learning rate.
        log_interval: Epochs between symmetry measurements.
        seed: Random seed for full reproducibility.
        modulus: Modulus for the addition task.

    Returns:
        Dictionary with training results including symmetry trajectory,
        loss trajectory, and final test accuracy.
    """
    torch.manual_seed(seed)

    input_dim = 2 * modulus
    output_dim = modulus

    model = SymmetricMLP(input_dim, hidden_dim, output_dim, epsilon, seed)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    x_train, y_train, x_test, y_test = generate_modular_addition_data(modulus, seed)

    # Record initial symmetry
    epochs_logged: List[int] = [0]
    symmetry_values: List[float] = [
        symmetry_metric(model.fc1.weight.data.clone())
    ]
    loss_values: List[float] = []
    train_acc_values: List[float] = []

    n_train = x_train.size(0)
    gen = torch.Generator()
    gen.manual_seed(seed)

    for epoch in range(1, num_epochs + 1):
        model.train()

        # Shuffle training data each epoch (source of SGD noise)
        perm = torch.randperm(n_train, generator=gen)
        x_shuffled = x_train[perm]
        y_shuffled = y_train[perm]

        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            x_batch = x_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches

        if epoch % log_interval == 0:
            epochs_logged.append(epoch)
            sym = symmetry_metric(model.fc1.weight.data.clone())
            symmetry_values.append(sym)
            loss_values.append(avg_loss)

            # Training accuracy
            model.eval()
            with torch.no_grad():
                train_pred = model(x_train).argmax(dim=1)
                train_acc = (train_pred == y_train).float().mean().item()
            train_acc_values.append(train_acc)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        test_pred = model(x_test).argmax(dim=1)
        test_acc = (test_pred == y_test).float().mean().item()

        train_pred = model(x_train).argmax(dim=1)
        final_train_acc = (train_pred == y_train).float().mean().item()

    return {
        "hidden_dim": hidden_dim,
        "epsilon": epsilon,
        "seed": seed,
        "epochs_logged": epochs_logged,
        "symmetry_values": symmetry_values,
        "loss_values": loss_values,
        "train_acc_values": train_acc_values,
        "final_test_acc": test_acc,
        "final_train_acc": final_train_acc,
        "initial_symmetry": symmetry_values[0],
        "final_symmetry": symmetry_values[-1],
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "modulus": modulus,
    }


def run_all_experiments(
    hidden_dims: List[int] = None,
    epsilons: List[float] = None,
    num_epochs: int = 2000,
    batch_size: int = 256,
    lr: float = 0.1,
    log_interval: int = 50,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Run the full grid of experiments.

    Args:
        hidden_dims: List of hidden layer widths to test.
        epsilons: List of perturbation scales to test.
        num_epochs: Number of training epochs per run.
        batch_size: Mini-batch size.
        lr: Learning rate.
        log_interval: Epochs between measurements.
        seed: Base random seed.

    Returns:
        List of result dictionaries, one per (hidden_dim, epsilon) pair.
    """
    if hidden_dims is None:
        hidden_dims = [16, 32, 64, 128]
    if epsilons is None:
        epsilons = [0.0, 1e-6, 1e-4, 1e-2, 1e-1]

    results = []
    total = len(hidden_dims) * len(epsilons)
    run_idx = 0

    for hd in hidden_dims:
        for eps in epsilons:
            run_idx += 1
            print(
                f"  [{run_idx}/{total}] hidden_dim={hd}, epsilon={eps:.1e} ...",
                end="",
                flush=True,
            )
            result = train_single_run(
                hidden_dim=hd,
                epsilon=eps,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=lr,
                log_interval=log_interval,
                seed=seed,
            )
            print(
                f" sym: {result['initial_symmetry']:.4f} -> {result['final_symmetry']:.4f}, "
                f"test_acc: {result['final_test_acc']:.4f}"
            )
            results.append(result)

    return results
