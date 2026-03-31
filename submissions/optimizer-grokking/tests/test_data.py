"""Tests for modular arithmetic data generation and splitting."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data import generate_all_pairs, split_data, make_loaders, PRIME, SEED


def test_generate_all_pairs_count():
    """All p^2 pairs should be generated."""
    pairs = generate_all_pairs(p=PRIME)
    assert len(pairs) == PRIME * PRIME


def test_generate_all_pairs_correctness():
    """Each triple should satisfy (a + b) mod p == target."""
    pairs = generate_all_pairs(p=PRIME)
    for a, b, target in pairs:
        assert (a + b) % PRIME == target


def test_generate_all_pairs_small():
    """Verify with a small prime."""
    pairs = generate_all_pairs(p=5)
    assert len(pairs) == 25
    assert (0, 0, 0) in pairs
    assert (2, 3, 0) in pairs  # (2+3) mod 5 = 0
    assert (4, 4, 3) in pairs  # (4+4) mod 5 = 3


def test_split_data_sizes():
    """Train/test split should have correct sizes."""
    train_ds, test_ds = split_data(p=PRIME)
    total = PRIME * PRIME
    n_train = int(total * 0.7)
    assert len(train_ds) == n_train
    assert len(test_ds) == total - n_train


def test_split_data_no_overlap():
    """Train and test sets should not overlap."""
    train_ds, test_ds = split_data(p=PRIME)
    train_set = set()
    for i in range(len(train_ds)):
        a, b, t = train_ds[i]
        train_set.add((a.item(), b.item()))

    for i in range(len(test_ds)):
        a, b, t = test_ds[i]
        pair = (a.item(), b.item())
        assert pair not in train_set, f"Overlap found: {pair}"


def test_split_reproducibility():
    """Same seed should produce same split."""
    train1, test1 = split_data(p=PRIME, seed=SEED)
    train2, test2 = split_data(p=PRIME, seed=SEED)
    assert len(train1) == len(train2)
    for i in range(len(train1)):
        assert train1[i][0].item() == train2[i][0].item()
        assert train1[i][1].item() == train2[i][1].item()


def test_make_loaders():
    """DataLoaders should be iterable and yield correct shapes."""
    train_loader, test_loader = make_loaders(p=PRIME, batch_size=128)
    batch = next(iter(train_loader))
    assert len(batch) == 3  # a, b, targets
    a, b, targets = batch
    assert a.shape[0] <= 128
    assert b.shape[0] <= 128
    assert targets.shape[0] <= 128
