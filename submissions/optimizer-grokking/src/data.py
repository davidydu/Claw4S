"""Modular arithmetic dataset for grokking experiments.

Generates (a, b) -> (a + b) mod p for all pairs 0 <= a, b < p.
Splits into train/test with a fixed seed for reproducibility.
"""

import torch
from torch.utils.data import Dataset, DataLoader

PRIME = 97
TRAIN_FRACTION = 0.7
SEED = 42


class ModularAdditionDataset(Dataset):
    """Dataset of (a, b) -> (a + b) mod p."""

    def __init__(self, pairs: list[tuple[int, int, int]]):
        self.a = torch.tensor([p[0] for p in pairs], dtype=torch.long)
        self.b = torch.tensor([p[1] for p in pairs], dtype=torch.long)
        self.targets = torch.tensor([p[2] for p in pairs], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.a[idx], self.b[idx], self.targets[idx]


def generate_all_pairs(p: int = PRIME) -> list[tuple[int, int, int]]:
    """Generate all (a, b, (a+b) mod p) triples."""
    pairs = []
    for a in range(p):
        for b in range(p):
            pairs.append((a, b, (a + b) % p))
    return pairs


def split_data(
    p: int = PRIME,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = SEED,
) -> tuple[ModularAdditionDataset, ModularAdditionDataset]:
    """Split all pairs into train and test sets deterministically.

    Returns:
        (train_dataset, test_dataset)
    """
    all_pairs = generate_all_pairs(p)
    n = len(all_pairs)
    n_train = int(n * train_fraction)

    generator = torch.Generator()
    generator.manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    train_pairs = [all_pairs[i] for i in indices[:n_train]]
    test_pairs = [all_pairs[i] for i in indices[n_train:]]

    return ModularAdditionDataset(train_pairs), ModularAdditionDataset(test_pairs)


def make_loaders(
    p: int = PRIME,
    train_fraction: float = TRAIN_FRACTION,
    seed: int = SEED,
    batch_size: int = 512,
) -> tuple[DataLoader, DataLoader]:
    """Create train and test DataLoaders.

    Returns:
        (train_loader, test_loader)
    """
    train_ds, test_ds = split_data(p, train_fraction, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
