"""Activation sparsity metrics.

Computes dead neuron fraction, near-dead neuron fraction, activation
entropy, and mean activation magnitude from post-ReLU hidden activations.
"""

import torch


def dead_neuron_fraction(activations: torch.Tensor, threshold: float = 0.0) -> float:
    """Fraction of neurons that are always zero across the batch.

    A neuron is "dead" if its activation is <= threshold for every
    sample in the batch.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch_size, hidden_dim). Post-ReLU activations.
    threshold : float
        Activation values <= this are considered zero. Default 0.0.

    Returns
    -------
    float
        Fraction in [0, 1]. Higher means more dead neurons.
    """
    if activations.numel() == 0:
        return 0.0
    # For each neuron, check if max activation across batch is <= threshold
    max_per_neuron = activations.max(dim=0).values  # shape (hidden_dim,)
    n_dead = (max_per_neuron <= threshold).sum().item()
    n_total = activations.shape[1]
    return n_dead / n_total


def near_dead_fraction(activations: torch.Tensor, threshold: float = 1e-3) -> float:
    """Fraction of neurons with mean activation below a small threshold.

    A softer version of dead_neuron_fraction that captures neurons
    that are "nearly dead" -- still technically active but contributing
    negligibly to the network output.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch_size, hidden_dim). Post-ReLU activations.
    threshold : float
        Neurons with mean activation below this are "near-dead".

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if activations.numel() == 0:
        return 0.0
    mean_per_neuron = activations.mean(dim=0)  # shape (hidden_dim,)
    n_near_dead = (mean_per_neuron < threshold).sum().item()
    n_total = activations.shape[1]
    return n_near_dead / n_total


def zero_fraction(activations: torch.Tensor) -> float:
    """Fraction of all activation values that are exactly zero.

    This measures overall sparsity of the activation pattern, not
    per-neuron death. A higher value means more sparse activations.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch_size, hidden_dim). Post-ReLU activations.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if activations.numel() == 0:
        return 0.0
    return (activations == 0).float().mean().item()


def activation_entropy(activations: torch.Tensor, n_bins: int = 50) -> float:
    """Entropy of the activation distribution (discretized).

    Measures the diversity of activation values across the hidden layer.
    Low entropy means activations are concentrated (e.g., mostly zero).
    High entropy means activations are spread across many values.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch_size, hidden_dim). Post-ReLU activations.
    n_bins : int
        Number of histogram bins for discretization.

    Returns
    -------
    float
        Shannon entropy in nats. Non-negative.
    """
    flat = activations.flatten().float()
    if flat.numel() == 0:
        return 0.0

    # Build histogram
    min_val = flat.min().item()
    max_val = flat.max().item()
    if max_val <= min_val:
        return 0.0

    counts = torch.histc(flat, bins=n_bins, min=min_val, max=max_val)
    probs = counts / counts.sum()
    probs = probs[probs > 0]

    entropy = -(probs * probs.log()).sum().item()
    return entropy


def mean_activation_magnitude(activations: torch.Tensor) -> float:
    """Mean absolute activation value across all neurons and samples.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch_size, hidden_dim). Post-ReLU activations.

    Returns
    -------
    float
        Mean magnitude. For ReLU outputs, this equals the mean value
        since all activations are >= 0.
    """
    if activations.numel() == 0:
        return 0.0
    return activations.abs().mean().item()


def compute_all_metrics(activations: torch.Tensor) -> dict:
    """Compute all sparsity metrics for a batch of activations.

    Parameters
    ----------
    activations : torch.Tensor
        Shape (batch_size, hidden_dim). Post-ReLU activations.

    Returns
    -------
    dict
        Keys: 'dead_neuron_fraction', 'near_dead_fraction',
              'zero_fraction', 'activation_entropy',
              'mean_activation_magnitude'.
    """
    return {
        "dead_neuron_fraction": dead_neuron_fraction(activations),
        "near_dead_fraction": near_dead_fraction(activations),
        "zero_fraction": zero_fraction(activations),
        "activation_entropy": activation_entropy(activations),
        "mean_activation_magnitude": mean_activation_magnitude(activations),
    }
