"""Tests for pruning strategies."""

import torch
from src.model import TwoLayerMLP
from src.pruning import magnitude_prune, random_prune, structured_prune, apply_masks


def _make_model():
    torch.manual_seed(42)
    return TwoLayerMLP(input_dim=10, hidden_dim=32, output_dim=5)


def test_magnitude_prune_zero_sparsity():
    """Zero sparsity should not change the model."""
    model = _make_model()
    original = {n: p.clone() for n, p in model.named_parameters()}
    masks = magnitude_prune(model, 0.0)
    assert masks == {}
    for name, param in model.named_parameters():
        assert torch.equal(param, original[name])


def test_magnitude_prune_reduces_nonzero():
    """Magnitude pruning at 50% should roughly halve nonzero weights."""
    model = _make_model()
    total_before = model.count_nonzero_parameters()
    magnitude_prune(model, 0.5)
    nonzero_after = model.count_nonzero_parameters()
    # Allow some tolerance due to bias not being pruned
    assert nonzero_after < total_before * 0.7


def test_random_prune_reproducible():
    """Random pruning with same seed produces identical masks."""
    m1 = _make_model()
    m2 = _make_model()
    masks1 = random_prune(m1, 0.5, seed=42)
    masks2 = random_prune(m2, 0.5, seed=42)
    for key in masks1:
        assert torch.equal(masks1[key], masks2[key])


def test_random_prune_different_seeds():
    """Random pruning with different seeds produces different masks."""
    m1 = _make_model()
    m2 = _make_model()
    masks1 = random_prune(m1, 0.5, seed=42)
    masks2 = random_prune(m2, 0.5, seed=99)
    any_different = any(
        not torch.equal(masks1[k], masks2[k]) for k in masks1
    )
    assert any_different


def test_structured_prune_zeros_neurons():
    """Structured pruning should zero out entire rows in fc1."""
    model = _make_model()
    structured_prune(model, 0.5)
    # Check that some rows of fc1.weight are entirely zero
    row_norms = model.fc1.weight.data.norm(dim=1)
    zero_rows = (row_norms == 0).sum().item()
    assert zero_rows >= 10  # 50% of 32 = 16, allow some margin


def test_apply_masks_keeps_zeros():
    """After applying masks, pruned weights stay zero."""
    model = _make_model()
    masks = magnitude_prune(model, 0.5)

    # Simulate a gradient update that would change all params
    for p in model.parameters():
        p.data += 0.01 * torch.randn_like(p)

    # Re-apply masks
    apply_masks(model, masks)

    # Check that masked positions are zero
    for name, param in model.named_parameters():
        if name in masks:
            assert (param.data[masks[name] == 0] == 0).all()
