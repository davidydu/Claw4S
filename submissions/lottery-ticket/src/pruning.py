"""Pruning strategies for lottery ticket experiments.

Three pruning strategies applied at initialization:
1. Magnitude pruning: remove weights with smallest absolute values
2. Random pruning: remove weights uniformly at random
3. Structured pruning: remove entire hidden neurons by L2 norm
"""

import torch
import torch.nn as nn
import numpy as np


def magnitude_prune(model: nn.Module, sparsity: float, seed: int = 42) -> dict:
    """Prune weights by global magnitude at initialization.

    Zeros out the smallest `sparsity` fraction of all weight parameters
    (biases are left untouched).

    Args:
        model: The neural network to prune.
        sparsity: Fraction of weights to zero out (0.0 = no pruning, 0.9 = 90% pruned).
        seed: Random seed (unused here, included for API consistency).

    Returns:
        Dictionary with mask tensors keyed by parameter name.
    """
    if sparsity <= 0.0:
        return {}

    # Collect all weight magnitudes (exclude biases)
    all_weights = []
    weight_names = []
    for name, param in model.named_parameters():
        if "weight" in name:
            all_weights.append(param.data.abs().flatten())
            weight_names.append(name)

    all_magnitudes = torch.cat(all_weights)
    num_to_prune = int(sparsity * all_magnitudes.numel())

    if num_to_prune == 0:
        return {}

    threshold = torch.sort(all_magnitudes)[0][num_to_prune - 1]

    masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            mask = (param.data.abs() > threshold).float()
            param.data *= mask
            masks[name] = mask

    return masks


def random_prune(model: nn.Module, sparsity: float, seed: int = 42) -> dict:
    """Prune weights uniformly at random at initialization.

    Args:
        model: The neural network to prune.
        sparsity: Fraction of weights to zero out.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with mask tensors keyed by parameter name.
    """
    if sparsity <= 0.0:
        return {}

    rng = torch.Generator()
    rng.manual_seed(seed)

    masks = {}
    for name, param in model.named_parameters():
        if "weight" in name:
            mask = (torch.rand(param.shape, generator=rng) >= sparsity).float()
            param.data *= mask
            masks[name] = mask

    return masks


def structured_prune(model: nn.Module, sparsity: float, seed: int = 42) -> dict:
    """Prune entire hidden neurons by L2 norm of incoming weights.

    Removes the lowest-norm neurons from the first hidden layer,
    and zeros out corresponding rows in the second layer.

    Args:
        model: The neural network to prune (must have fc1 and fc2 attributes).
        sparsity: Fraction of hidden neurons to remove.
        seed: Random seed (unused here, included for API consistency).

    Returns:
        Dictionary with mask tensors keyed by parameter name.
    """
    if sparsity <= 0.0:
        return {}

    # Compute L2 norm of each hidden neuron's incoming weights
    fc1_weight = model.fc1.weight.data  # shape: [hidden_dim, input_dim]
    neuron_norms = fc1_weight.norm(dim=1)  # shape: [hidden_dim]

    hidden_dim = fc1_weight.shape[0]
    num_to_prune = int(sparsity * hidden_dim)

    if num_to_prune == 0:
        return {}
    if num_to_prune >= hidden_dim:
        num_to_prune = hidden_dim - 1  # keep at least 1 neuron

    # Find neurons to prune (smallest L2 norm)
    _, indices = torch.sort(neuron_norms)
    prune_indices = indices[:num_to_prune]

    # Create neuron-level mask
    neuron_mask = torch.ones(hidden_dim)
    neuron_mask[prune_indices] = 0.0

    masks = {}

    # Zero out pruned neurons in fc1 (rows) and fc1 bias
    fc1_mask = neuron_mask.unsqueeze(1).expand_as(model.fc1.weight.data)
    model.fc1.weight.data *= fc1_mask
    model.fc1.bias.data *= neuron_mask
    masks["fc1.weight"] = fc1_mask

    # Zero out pruned neurons in fc2 (columns)
    fc2_mask = neuron_mask.unsqueeze(0).expand_as(model.fc2.weight.data)
    model.fc2.weight.data *= fc2_mask
    masks["fc2.weight"] = fc2_mask

    return masks


def apply_masks(model: nn.Module, masks: dict) -> None:
    """Re-apply pruning masks after a gradient update.

    Called after each optimizer step to keep pruned weights at zero.

    Args:
        model: The neural network.
        masks: Dictionary of mask tensors from a pruning function.
    """
    for name, param in model.named_parameters():
        if name in masks:
            param.data *= masks[name]
